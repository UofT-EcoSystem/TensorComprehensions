/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tc/core/polyhedral/cuda/memory_promotion_heuristic.h"

#include <glog/logging.h>

#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/memory_promotion.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/polyhedral/unroll.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <type_traits>

namespace tc {
namespace polyhedral {
namespace {

/*
 * Is "id" a mapping of the type provided as template argument?
 */
template <typename MappingType>
bool isMappingIdType(const mapping::MappingId& id) {
  for (size_t i = 0; i < MappingType::kMaxDim; ++i) {
    if (id == MappingType::makeId(i)) {
      return true;
    }
  }
  return false;
}

/*
 * Is "tree" a mapping filter that maps identifiers of the type provided as
 * template argument?
 */
template <typename MappingType>
bool isMappingTo(const detail::ScheduleTree* tree) {
  using namespace detail;

  if (auto filterNode = tree->elemAs<ScheduleTreeElemMappingFilter>()) {
    for (auto& kvp : filterNode->mapping) {
      if (isMappingIdType<MappingType>(kvp.first)) {
        return true;
      }
    }
  }
  return false;
}

// Map global<->shared copy bands to threads, starting from the innermost
// loop as it iterates over the last subscript and will result in coalescing.
void mapCopiesToThreads(MappedScop& mscop, bool unroll) {
  using namespace detail;

  // Find all filters with reads from or writes to global memory.
  auto matcher = filter(
      [](isl::union_set uset) {
        auto sets = isl::UnionAsVector<isl::union_set>(uset);
        return std::all_of(sets.begin(), sets.end(), [](isl::set s) {
          auto readId = isl::id(s.get_ctx(), std::string(kReadIdName));
          auto writeId = isl::id(s.get_ctx(), std::string(kWriteIdName));
          return s.get_tuple_id() == readId || s.get_tuple_id() == writeId;
        });
      },
      any());

  auto root = mscop.scop().scheduleRoot();
  for (auto constNode : match(matcher, root)) {
    // We need to modify the nodes and have non-const mscop.
    auto node = const_cast<ScheduleTree*>(constNode);
    if (match(filter(band()), node).size() != 1) {
      std::stringstream ss;
      ss << "read/write filter not followed by a single band" << std::endl
         << *node;
      throw promotion::PromotionLogicError(ss.str());
    }

    auto bandNode = node->child({0});
    auto band = bandNode->elemAs<ScheduleTreeElemBand>();
    if (!band) {
      throw promotion::PromotionLogicError("no copy band");
    }

    // Check that we are not mapping to threads below other thread mappings.
    std::unordered_set<mapping::ThreadId, mapping::ThreadId::Hash> usedThreads;
    for (auto n : node->ancestors(root)) {
      if (isMappingTo<mapping::ThreadId>(n)) {
        throw promotion::PromotionBelowThreadsException(
            "attempted to map memory copies to threads below "
            "another thread mapping");
      }
    }

    mscop.mapThreadsBackward(bandNode);

    // Unroll if requested.
    if (unroll) {
      markUnroll(root, bandNode, mscop.unroll);
    }
  }
}

/*
 * Starting from the root, find all thread specific markers.  Use
 * DFSPreorder to make sure order is specified and consistent for tests.
 */
template <typename T>
std::vector<T> findThreadSpecificMarkers(T root) {
  using namespace tc::polyhedral::detail;
  static_assert(
      std::is_convertible<T, const ScheduleTree*>::value,
      "expecting ScheduleTree");

  return ScheduleTree::collectDFSPreorder(
      root, ScheduleTreeType::ThreadSpecificMarker);
}

/*
 * Return the thread specific markers in the tree rooted at "root"
 * that are relevant for "node".
 *
 * Every branch in the tree has exactly one thread marker.
 * If "node" appears underneath a thread marker, then return
 * that single thread marker.
 * Otherwise, return the (possibly multiple) thread markers
 * in the subtree rooted at "node".
 */
template <typename T>
std::vector<T> collectBranchMarkers(T root, T node) {
  using namespace detail;
  static_assert(
      std::is_convertible<T, const ScheduleTree*>::value,
      "expecting ScheduleTree");

  auto filterMarker = [](T tree) {
    return tree->type_ == ScheduleTreeType::ThreadSpecificMarker;
  };

  auto ancestors = node->ancestors(root);
  ancestors = functional::Filter(filterMarker, ancestors);
  if (ancestors.size() > 0) {
    return ancestors;
  }
  return findThreadSpecificMarkers(node);
}

/*
 * Transform schedule bands into a union_map.
 * Takes all partial schedules at leaves as MUPAs (without accounting for
 * intermediate non-band nodes), intersects
 * their domain with the filters between the root and the
 * current leaves and transforms them into union maps.
 * Mapping filters are ignored.
 */
isl::union_map fullSchedule(const detail::ScheduleTree* root) {
  using namespace tc::polyhedral::detail;

  if (!root->elemAs<ScheduleTreeElemDomain>()) {
    throw promotion::PromotionLogicError("expected root to be a domain node");
  }

  std::function<bool(const ScheduleTree* tree)> isLeaf =
      [](const ScheduleTree* tree) { return tree->numChildren() == 0; };

  // Find all innermost nodes.
  auto leaves = functional::Filter(isLeaf, ScheduleTree::collect(root));

  // Take a union of partial schedules of the innermost nodes.  Because they
  // are innermost, the partial schedule can no longer be affected by deeper
  // nodes and hence is full.
  auto schedule = isl::union_map::empty(
      root->elemAs<ScheduleTreeElemDomain>()->domain_.get_space());
  for (auto node : leaves) {
    auto domain = root->elemAs<ScheduleTreeElemDomain>()->domain_;
    auto prefixMupa = prefixScheduleMupa(root, node);
    if (auto band = node->elemAs<ScheduleTreeElemBand>()) {
      prefixMupa = prefixMupa.flat_range_product(band->mupa_);
    }

    auto pathToRoot = node->ancestors(root);
    pathToRoot.push_back(node);
    for (auto n : pathToRoot) {
      if (auto filterNode = n->elemAs<ScheduleTreeElemFilter>()) {
        domain = domain.intersect(filterNode->filter_);
      }
    }

    prefixMupa = prefixMupa.intersect_domain(domain);

    schedule = schedule.unite(isl::union_map::from(prefixMupa));
    if (!schedule.is_single_valued()) {
      std::stringstream ss;
      ss << "schedules must be single-valued " << schedule << std::endl
         << *root;
      throw promotion::PromotionLogicError(ss.str());
    }
  }
  return schedule;
}

/*
 * Check if a reference group features reuse within the "outer" schedule.
 * In particular, check that for some given point in the outer schedule and
 * some given group element, there is more than one statement instance
 * accessing the element within the point in the outer schedule.
 * In other words, check that the mapping from statement instances
 * to pairs of outer schedule points and group elements is not injective.
 */
bool hasReuseWithin(
    const TensorReferenceGroup& group,
    isl::multi_union_pw_aff outer) {
  auto map = isl::union_map::from(outer);
  map = map.range_product(group.originalAccesses());
  return !map.is_injective();
}

/*
 * Create a map that increments the "dim"-th dimension and keeps all other
 * dimensions unchanged.
 */
isl::map makeNextElementMap(isl::space setSpace, unsigned dim) {
  if (dim < 0 || dim >= setSpace.dim(isl::dim_type::set)) {
    std::stringstream ss;
    ss << dim << "  is out of [0, " << setSpace.dim(isl::dim_type::set)
       << ") range";
    throw promotion::OutOfRangeException(ss.str());
  }

  auto mapSpace = setSpace.map_from_set();
  auto identityMA = isl::multi_aff::identity(mapSpace);
  auto aff = identityMA.get_aff(dim);
  identityMA = identityMA.set_aff(dim, aff + 1);
  return isl::map(identityMA);
}

/*
 * Return the outermost thread mapping filter among the ancestors of "node",
 * assuming that there is at least one.
 */
const detail::ScheduleTree* findThreadMappingAncestor(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node) {
  auto ancestors = node->ancestors(root);
  ancestors = functional::Filter(isMappingTo<mapping::ThreadId>, ancestors);
  if (ancestors.size() < 1) {
    throw promotion::PromotionLogicError("missing MappingFilter");
  }
  return ancestors[0];
}

/*
 * Should this reference group be promoted for the purpose of coalescing?
 *
 * If the reference group is not already accessed in a coalesced way,
 * then the group should be promoted.
 * If a branch is mapped to a single thread, then the accesses
 * in that branch are not considered to contribute to the usefulness
 * of promoting.
 *
 * The check for coalesced accesses is performed as follows.
 * Check if incrementing the schedule dimension mapped to
 * Thread::x results in the last tensor index being incremented as well.
 * Since accesses in the group may belong to different statements, which may
 * have different loops mapped to Thread::x, perform the check for each thread
 * mapping on the statements active at "node" (either a single ancestor,
 * or one or more descendants).
 * The iteration over the spaces is used to handle the case where
 * one of the subbranches does not access the tensor and
 * the scheduled accesses are empty.  The group is
 * accessed in a coalesced way if all references in this group are accessed in
 * a coalesced way.
 */
bool promotionImprovesCoalescing(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node,
    const TensorReferenceGroup& group,
    isl::union_map schedule) {
  auto originalAccesses = group.originalAccesses();

  auto markers = collectBranchMarkers(root, node);
  for (auto marker : markers) {
    auto mapping = findThreadMappingAncestor(root, marker);
    size_t nMappedThreads = marker->scheduleDepth(mapping);
    if (nMappedThreads == 0) {
      continue;
    }
    auto depth = marker->scheduleDepth(root);
    auto activePoints = activeDomainPoints(root, mapping);
    auto localAccesses = originalAccesses.intersect_domain(activePoints);
    auto scheduledAccesses = localAccesses.apply_domain(schedule);
    for (auto access : isl::UnionAsVector<isl::union_map>(scheduledAccesses)) {
      auto scheduleSpace = access.get_space().domain();
      auto tensorSpace = access.get_space().range();
      auto elementToNext = makeNextElementMap(
          tensorSpace, tensorSpace.dim(isl::dim_type::set) - 1);
      auto scheduleToNextX = makeNextElementMap(scheduleSpace, depth - 1);
      auto accessedByAdjacentX =
          scheduleToNextX.apply_domain(access).apply_range(access);

      if (not accessedByAdjacentX.is_subset(elementToNext)) {
        return true;
      }
    }
  }
  return false;
}

/*
 * Starting from the root, find bands where depth is reached.  Using
 * DFSPreorder to make sure order is specified and consistent for tests.
 */
std::vector<detail::ScheduleTree*> bandsContainingScheduleDepth(
    detail::ScheduleTree* root,
    size_t depth) {
  using namespace tc::polyhedral::detail;

  auto bands =
      ScheduleTree::collectDFSPreorder(root, detail::ScheduleTreeType::Band);
  std::function<bool(ScheduleTree * st)> containsDepth = [&](ScheduleTree* st) {
    auto depthBefore = st->scheduleDepth(root);
    auto band = st->elemAs<ScheduleTreeElemBand>();
    auto depthAfter = depthBefore + band->nMember();
    return depthBefore < depth && depthAfter >= depth;
  };
  return functional::Filter(containsDepth, bands);
}

/*
 * Split bands so that the "depth"-th dimension is always the last in some
 * band.  Return such bands.
 */
std::vector<detail::ScheduleTree*> bandsSplitAfterDepth(
    const std::vector<detail::ScheduleTree*>& bands,
    detail::ScheduleTree* root,
    size_t depth) {
  using namespace tc::polyhedral::detail;

  std::function<ScheduleTree*(ScheduleTree*)> splitAtDepth =
      [&](ScheduleTree* st) {
        auto nMember = st->elemAs<ScheduleTreeElemBand>()->nMember();
        auto scheduleDepth = st->scheduleDepth(root);
        auto depthAfter = scheduleDepth + nMember;
        return depthAfter == depth ? st
                                   : bandSplit(root, st, depth - scheduleDepth);
      };
  return functional::Map(splitAtDepth, bands);
}

/*
 * For every place in the schedule tree where schedule depth (i.e., the number
 * of preceding band members) is "depth", promote tensor reference groups to
 * shared memory.  Split bands if necessary to insert promotions.
 *
 * Use at most "maxMemory" bytes.  If a groups does not fit the remaining
 * memory, do not promote it and keep looking for a smaller group.
 *
 * Only promote if the tensor elements referenced by the group are reused or
 * accessed in a non-coalesced way.
 */
void promoteToSharedGreedy(
    Scop& scop,
    const Block& block,
    size_t depth,
    size_t maxMemory) {
  using namespace tc::polyhedral::detail;

  if (depth == 0) {
    throw promotion::PromotionNYI("promotion before any band");
  }

  auto root = scop.scheduleRoot();

  // 1. Collect all bands with a member located at the given depth in the
  // overall schedule.  Make sure this is the last member of the band by
  // splitting off the subsequent members into a different band.
  auto bands = bandsContainingScheduleDepth(root, depth);
  bands = bandsSplitAfterDepth(bands, root, depth);

  // 2. Compute full schedule without mapping filters.  The filters would make
  // it impossible to test for coalescing by incrementing a member of a band as
  // only the values divisible by grid or block size pass through the filter.
  auto fullSched = fullSchedule(root);

  // 3. For each band that ends at "depth", take decisions about promotion
  // immediately below it in the tree.  In particular, promote if the
  // approximated footprint fits into the remaining memory, and the reference
  // group either features reuse or is accessed in a non-coalesced way, or
  // both.
  size_t remainingMemory = maxMemory;
  for (auto bandNode : bands) {
    auto activePoints = activeDomainPoints(root, bandNode);
    auto partialSched = partialSchedule(root, bandNode);

    auto groupMap = TensorReferenceGroup::accessedWithin(
        partialSched.intersect_domain(activePoints), scop.reads, scop.writes);
    // Pure affine schedule without (mapping) filters.
    auto partialSchedMupa = partialScheduleMupa(root, bandNode);

    // Prepare groups for sorting, to have specified order necessary for
    // reproducibility and tests.
    using TensorGroupList = std::pair<isl::id, TensorGroupsInfo>;
    std::vector<TensorGroupList> groupLists(
        std::make_move_iterator(groupMap.begin()),
        std::make_move_iterator(groupMap.end()));

    // Computes the total number of references in all groups.
    auto refsCount = [](const TensorGroupsInfo& info) {
      size_t refs = 0;
      for (auto const& group : info) {
        refs += group->referenceIds().size();
      }
      return refs;
    };

    // Sort by the total number of references, then by name.  Because names are
    // guarenteed to be unique, the order is total.
    std::sort(
        groupLists.begin(),
        groupLists.end(),
        [refsCount](const TensorGroupList& l1, const TensorGroupList& l2) {
          auto r1 = refsCount(l1.second);
          auto r2 = refsCount(l2.second);
          return r1 == r2 ? l1.first.get_name() < l2.first.get_name() : r1 < r2;
        });
    for (auto& tensorGroups : groupLists) {
      auto tensorId = tensorGroups.first;
      // Sort the reference groups to prioritize groups with more references as
      // they are more likely to benefit from promotion.
      std::sort(
          tensorGroups.second.begin(),
          tensorGroups.second.end(),
          [refsCount](
              const std::unique_ptr<TensorReferenceGroup>& group1,
              const std::unique_ptr<TensorReferenceGroup>& group2) {
            return group1->referenceIds().size() >
                group2->referenceIds().size();
          });

      for (auto& group : tensorGroups.second) {
        auto sizes = group->approximationSizes();
        if (sizes.size() == 0) {
          throw promotion::PromotionLogicError("cannot promote a scalar");
        }
        if (sizes.back() % 2 == 0) {
          sizes.back() += 1;
        }
        auto nApproximationElements = std::accumulate(
            sizes.begin(), sizes.end(), 1, std::multiplies<size_t>());
        size_t memoryRequirement =
            nApproximationElements * scop.findArgument(tensorId).type().bytes();
        if (memoryRequirement > remainingMemory) {
          continue;
        }
        // Do not promote if the group features no reuse and is accessed in a
        // coalesced way.
        if (!hasReuseWithin(*group, partialSchedMupa) &&
            !promotionImprovesCoalescing(root, bandNode, *group, fullSched)) {
          continue;
        }

        scop.promoteGroup(
            Scop::PromotedDecl::Kind::SharedMem,
            tensorId,
            std::move(group),
            bandNode,
            partialSched,
            true);
        remainingMemory -= memoryRequirement;
      }
    }
    scop.insertSyncsAroundCopies(bandNode);
  }
}

// Returns a union of all mapping filters to "MappingType" in "scop".
template <typename MappingType>
isl::union_set collectMappingsTo(const Scop& scop) {
  auto root = scop.scheduleRoot();
  auto domain = scop.domain();
  auto mappingFilters = detail::ScheduleTree::collect(
      root, detail::ScheduleTreeType::MappingFilter);
  mappingFilters = functional::Filter(isMappingTo<MappingType>, mappingFilters);
  for (auto mf : mappingFilters) {
    auto filterNode = mf->elemAs<detail::ScheduleTreeElemMappingFilter>();
    domain = domain.intersect(filterNode->filter_);
  }
  return domain;
}

/*
 * Check that accesses described by "group" are thread-private within the given
 * scope, i.e. that no two threads access the same tensor element.  The scope
 * is defined by "prefixSchedule" that describes the outer loops.  Within
 * different instances of "prefixSchedule", threads may access the same
 * element.  It is expected to be copied to registers at the beginning of the
 * scope, and back to some location accessible to all threads at the end of the
 * scope.  If synchronization is necessary due to data dependences, it should
 * have been inserted by the mapping at the end of the scope.
 */
bool isThreadPrivate(
    const TensorReferenceGroup& group,
    isl::multi_union_pw_aff prefixSchedule,
    isl::multi_union_pw_aff threadSchedule) {
  auto accesses = group.originalAccesses(); // D -> A
  auto prefixScheduleUmap = isl::union_map::from(prefixSchedule); // D -> Sp
  auto accessesWithinScope =
      accesses.range_product(prefixScheduleUmap); // D -> (Sp -> A)
  auto threadScheduleUmap = isl::union_map::from(threadSchedule); // D -> St
  auto threadAccesses =
      accessesWithinScope.apply_domain(threadScheduleUmap); // T -> (Sp -> A)
  return threadAccesses.is_injective();
}

/*
 * Check if the band was mapped to threads.
 * We currently map entire bands to threads and insert a thread-specific marker
 * immediately after such bands.
 */
inline bool isThreadMappedBand(const detail::ScheduleTree* tree) {
  return matchOne(band(threadSpecific(any())), tree);
}

/*
 * Check that only unrolled loops may appear in access subscripts.
 * Because the scoping point can be above a branching tree, descend into each
 * leaf of the subtree below the scoping point.  For each leaf, construct an
 * affine multi-expression containing only those band members between the
 * scoping point and the leaf that are fully unrolled.  If band members are
 * mapped to threads, do not take into account the parts that will appear in
 * subscripts as thread identifiers, i.e. subtract the mapped affine functions
 * from the schedule affine functions.
 *
 * Within each instance of the scope, check that loops that are either unrolled
 * or mapped to threads access a single tensor element in the group (other loop
 * indices will then not appear in the subscripts, making register promotion
 * possible).  In other words, check that the relation between the flat product
 * of prefix, thread-mapped, and unrolled loop indices and accessed elements is
 * single-valued.
 * TODO: note that if a group is formed from partially overlapping references,
 * one must consider per-reference access relation for single-valuedness as
 * different references may have different values, but all of them remain
 * independent of non-unrolled loop iterators.
 */
bool accessSubscriptsAreUnrolledLoops(
    const TensorReferenceGroup& group,
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* scope,
    isl::multi_union_pw_aff threadSchedule) {
  using namespace detail;

  auto prefixSchedule = partialScheduleMupa(root, scope);

  auto nodes = ScheduleTree::collect(scope);
  auto leaves = functional::Filter(
      [](const ScheduleTree* tree) { return tree->numChildren() == 0; }, nodes);

  auto domainNode = root->elemAs<detail::ScheduleTreeElemDomain>();
  CHECK(domainNode);
  auto domain = domainNode->domain_;

  // Descend into every leaf.
  for (auto leaf : leaves) {
    auto ancestors = leaf->ancestors(root);
    ancestors.push_back(leaf);
    auto subdomain = activeDomainPointsBelow(root, leaf);

    auto unrolledDims = isl::union_pw_aff_list(leaf->ctx_, 1);
    for (auto node : ancestors) {
      auto band = node->elemAs<detail::ScheduleTreeElemBand>();
      if (!band) {
        continue;
      }

      isl::multi_union_pw_aff schedule = band->mupa_;
      if (isThreadMappedBand(node)) {
        // Band members are mapped to threads in inverse order.  There may be
        // less members than thread dimensions (the remaining thread dimensions
        // are mapped to 0).  Transform the thread schedule accordingly before
        // subtracting it from the band schedule.
        auto bandCompatibleThreadSchedule =
            isl::union_pw_aff_list(schedule.get_ctx(), band->nMember());
        auto threadScheduleDims =
            threadSchedule.get_union_pw_aff_list().reverse();
        CHECK_LE(band->nMember(), static_cast<size_t>(threadScheduleDims.n()));
        for (int i = 0; i < band->nMember(); ++i) {
          bandCompatibleThreadSchedule =
              bandCompatibleThreadSchedule.add(threadScheduleDims.get(i));
        }

        schedule = schedule.sub(isl::multi_union_pw_aff(
            schedule.get_space(), bandCompatibleThreadSchedule));
      }
      schedule = schedule.intersect_domain(subdomain);

      for (size_t i = 0, e = band->nMember(); i < e; ++i) {
        if (!band->unroll_[i]) {
          continue;
        }
        unrolledDims = unrolledDims.add(schedule.get_union_pw_aff(i));
      }
    }

    auto space = isl::space(leaf->ctx_, 0, unrolledDims.n())
                     .align_params(subdomain.get_space());
    auto unrolledDimsMupa = isl::multi_union_pw_aff(space, unrolledDims);

    // It is possible that no loops are unrolled, in which case
    // unrolledDimsMupa is zero-dimensional and needs an explicit domain
    // to be convertible to a union_map.
    unrolledDimsMupa =
        unrolledDimsMupa.intersect_domain(group.originalAccesses().domain());

    auto accesses = group.originalAccesses();
    auto schedule = prefixSchedule.flat_range_product(threadSchedule)
                        .flat_range_product(unrolledDimsMupa);
    accesses = accesses.apply_domain(isl::union_map::from(schedule));

    if (!accesses.is_single_valued()) {
      return false;
    }
  }

  return true;
}

/*
 * Check if the tensor reference group "group" can be promoted to registers at
 * the given "scope" in the schedule tree of "scop".
 * In particular, check that no elements in this group are accessed by
 * different threads and that all loop indices that may appear in subscripts
 * correspond to unrolled loops.
 */
bool isPromotableToRegistersBelow(
    const TensorReferenceGroup& group,
    const MappedScop& scop,
    const detail::ScheduleTree* scope) {
  auto prefixSchedule = partialScheduleMupa(scop.schedule(), scope);
  auto threadSchedule = scop.threadMappingSchedule(scop.schedule());
  return isThreadPrivate(group, prefixSchedule, threadSchedule) &&
      accessSubscriptsAreUnrolledLoops(
             group, scop.schedule(), scope, threadSchedule);
}
} // namespace

void promoteGreedilyAtDepth(
    MappedScop& mscop,
    size_t depth,
    size_t sharedMemorySize,
    bool unrollCopies) {
  // 1. Promote using heuristic.
  promoteToSharedGreedy(
      mscop.scop(), mscop.numThreads, depth, sharedMemorySize);

  // 2. Map copies to shared, state by copy
  mapCopiesToThreads(mscop, unrollCopies);
}

void promoteToRegistersBelow(MappedScop& mscop, detail::ScheduleTree* scope) {
  auto& scop = mscop.scop();

  // Compute groups specific to threads and block by including the mappings
  // into the domain of the partials schedule.
  auto mapping = collectMappingsTo<mapping::ThreadId>(scop).intersect(
      collectMappingsTo<mapping::BlockId>(scop));
  auto schedule = partialSchedule(scop.scheduleRoot(), scope);
  auto groupMap = TensorReferenceGroup::accessedWithin(
      schedule.intersect_domain(mapping), scop.reads, scop.writes);

  // FIXME: this schedule is also computed in
  // isPromotableToRegistersBelow
  auto threadSchedule = mscop.threadMappingSchedule(mscop.schedule());

  for (auto& tensorGroups : groupMap) {
    auto tensorId = tensorGroups.first;

    // TODO: sort using heuristic and count the total number of registers

    // FIXME: this schedule is also computed in
    // isPromotableToRegistersBelow
    auto partialSched = partialScheduleMupa(scop.scheduleRoot(), scope);
    auto partialSchedUM = partialSchedule(scop.scheduleRoot(), scope);

    for (auto& group : tensorGroups.second) {
      auto sizes = group->approximationSizes();
      // No point in promoting a scalar that will go to a register anyway.
      if (sizes.size() == 0) {
        continue;
      }
      if (!isPromotableToRegistersBelow(*group, mscop, scope)) {
        continue;
      }
      // Check reuse within threads.
      auto schedule = partialSched.flat_range_product(threadSchedule);
      if (!hasReuseWithin(*group, schedule)) {
        continue;
      }

      // TODO: if something is already in shared, but reuse it within one
      // thread only, there is no point in keeping it in shared _if_ it
      // gets promoted into a register.

      scop.promoteGroup(
          Scop::PromotedDecl::Kind::Register,
          tensorId,
          std::move(group),
          scope,
          partialSchedUM);
    }
  }
}

// Promote at the positions of the thread specific markers.
void promoteToRegistersBelowThreads(MappedScop& mscop, size_t nRegisters) {
  using namespace tc::polyhedral::detail;

  auto root = mscop.scop().scheduleRoot();
  auto markers = findThreadSpecificMarkers(root);

  for (auto marker : markers) {
    promoteToRegistersBelow(mscop, marker);
  }
}

} // namespace polyhedral
} // namespace tc
