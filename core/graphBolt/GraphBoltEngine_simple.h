// Copyright (c) 2020 Mugilan Mariappan, Joanna Che and Keval Vora.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GRAPHBOLT_ENGINE_SIMPLE_H
#define GRAPHBOLT_ENGINE_SIMPLE_H

#include "GraphBoltEngine.h"


/*
 * Profiling points
 *
 * We borrowed profiling system from LegoOS.
 * LegoOS: https://github.com/WukLab/LegoOS
 * 
 * We added profiling outside of kernel itself.
 * For example, kernel module which is not compiled with 
 * the kernel can use profiling points defined in the kernel
 * and use PROFILE_LEAVE_PTR instead of PROFILE_LEAVE.
 * Those profiling points used outside of the kernel
 * should be exported in this file by using 
 * PROTO_PROFILE_WITH_EXPORT()
 */


// .h
#include <string>
#include <atomic>
#include <chrono>

#define CONFIG_PROFILE_POINTS
#define MAX_PROFILE_POINTS 32
#define MAX_PROFILE_THREADS 16
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif
#ifndef CACHE_SIZE
#define CACHE_SIZE (PAGE_SIZE * 16)
#endif

enum {
  PP_MUTEX_ACQ = 0,
  PP_MUTEX_RLX,
	PP_MUTEX_CS,
	NUM_PP
};

struct profile_point {
    unsigned long nr;
    double time_us;
};

struct atomic_profile_point {
	std::atomic_ulong nr;
	std::atomic<double> time_us;
};

struct alignas(PAGE_SIZE) profile_point_arr {
	struct profile_point arr[MAX_PROFILE_POINTS];
};

struct alignas(PAGE_SIZE) atomic_profile_point_arr {
	struct atomic_profile_point arr[MAX_PROFILE_POINTS];
};

void print_local_profile_points(void);
void print_global_profile_points(void);
void clear_local_profile_points(void);
void report_local_profile_points(void);
void profile_add(int pp, double time_us);

#define _PP_TIME(pp, var)	__##pp##var

#define PROFILE_START(pp)             \
	auto _PP_TIME(pp, t_start) = std::chrono::high_resolution_clock::now();

#define PROFILE_LEAVE(pp)		\
		auto _PP_TIME(pp, t_end) = std::chrono::high_resolution_clock::now();                                \
		std::chrono::duration<double, std::micro> _PP_TIME(pp, t_double) = _PP_TIME(pp, t_end) - _PP_TIME(pp, t_start);                              \
		profile_add(pp, _PP_TIME(pp, t_double).count());


// .c
std::string pp_names[NUM_PP] = {
  "mutex_acquisition",
  "mutex_release",
  "critical section"
};

thread_local struct profile_point_arr pps;
struct atomic_profile_point_arr pps_global;

void print_global_profile_points(void) {
    printf("--- global profile points ---\n");
    for (int pp = 0; pp < NUM_PP; ++pp) {
        printf("%s nr[%lu] total[%lfus] avg[%lfus]\n", pp_names[pp].c_str(),
          pps_global.arr[pp].nr.load(), pps_global.arr[pp].time_us.load(),
          pps_global.arr[pp].time_us.load() / pps_global.arr[pp].nr.load());
    }
}

void print_local_profile_points(void) {
    printf("--- local profile points ---\n");
    for (int pp = 0; pp < NUM_PP; ++pp) {
        printf("%s nr[%lu] total[%lfus] avg[%lfus]\n", pp_names[pp].c_str(),
          pps.arr[pp].nr, pps.arr[pp].time_us,
          pps.arr[pp].time_us / pps.arr[pp].nr);
    }
}

void clear_local_profile_points(void) {
  memset(&pps, 0, sizeof(pps));
}

void report_local_profile_points(void) {
  for (int pp = 0; pp < NUM_PP; ++pp) {
    auto &_pp = pps_global.arr[pp];
    _pp.nr.fetch_add(pps.arr[pp].nr);
    for (double n = _pp.time_us.load(); !_pp.time_us.compare_exchange_strong(n, n + pps.arr[pp].time_us);)
      ;
  }
}

void profile_add(int pp, double time_us) {
    ++(pps.arr[pp].nr);
    pps.arr[pp].time_us += time_us;
}



// ======================================================================
// GRAPHBOLTENGINESIMPLE
// ======================================================================
template <class vertex, class AggregationValueType, class VertexValueType,
          class GlobalInfoType>
class GraphBoltEngineSimple
    : public GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                             GlobalInfoType> {
public:
  GraphBoltEngineSimple(graph<vertex> &_my_graph, int _max_iter,
                        GlobalInfoType &_static_data, bool _use_lock,
                        commandLine _config)
      : GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>(_my_graph, _max_iter, _static_data,
                                        _use_lock, _config) {
    use_source_contribution = true;
  }

  // ======================================================================
  // TEMPORARY STRUCTURES USED BY THE SIMPLE ENGINE
  // ======================================================================
  void createTemporaryStructures() {
    GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                    GlobalInfoType>::createTemporaryStructures();
  }
  void resizeTemporaryStructures() {
    GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                    GlobalInfoType>::resizeTemporaryStructures();
    initTemporaryStructures(n_old, n);
  }
  void freeTemporaryStructures() {
    GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                    GlobalInfoType>::freeTemporaryStructures();
  }
  void initTemporaryStructures() { initTemporaryStructures(0, n); }
  void initTemporaryStructures(long start_index, long end_index) {
    GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                    GlobalInfoType>::initTemporaryStructures(start_index,
                                                             end_index);
  }


  #define MIND_WARMUP_ITER 1

  // ======================================================================
  // TRADITIONAL INCREMENTAL COMPUTATION
  // ======================================================================
  // TODO : Currently, max_iterations = history_iterations.
  // Need to implement computation without history.
  int traditionalIncrementalComputation(int start_iteration) {
    timer iteration_timer, phase_timer;
    double misc_time, copy_time, phase_time, iteration_time;

    vertexSubset frontier_curr_vs(n, frontier_curr);
    bool use_delta = true;
    int iter = start_iteration;

    timer mind_timer;

    if (frontier_curr_vs.numNonzeros() == 0) {
      converged_iteration = start_iteration;

    } else {
      for (iter = start_iteration; iter < max_iterations; iter++) {
        // initialize timers
        {
          iteration_timer.start();
          phase_timer.start();
          misc_time = 0;
          copy_time = 0;
        }

        {
          if (iter == MIND_WARMUP_ITER)
            mind_timer.start();
        }

        // ========== COPY - Prepare curr iteration ==========
        if (iter > 0) {
          // Copy the aggregate and actual value from iter-1 to iter
          parallel_for(uintV v = 0; v < n; v++) {
            vertex_values[iter][v] = vertex_values[iter - 1][v];
            aggregation_values[iter][v] = aggregation_values[iter - 1][v];
            delta[v] = aggregationValueIdentity<AggregationValueType>();
          }
        }
        use_delta = shouldUseDelta(iter);

        // ========== MISC - count active edges for AE ==========
        phase_time = phase_timer.next();
        adaptive_executor.updateCopyTime(iter, phase_time);
        adaptive_executor.updateEdgesProcessed(iter, my_graph,
                                               frontier_curr_vs);
        misc_time = phase_timer.next();
        adaptive_executor.updateMiscTime(iter, phase_timer.next());

        // ========== EDGE COMPUTATION ==========
        if ((use_source_contribution) && (iter == 1)) {
          // Compute source contribution for first iteration
          parallel_for(uintV u = 0; u < n; u++) {
            if (frontier_curr[u]) {
              // compute source change in contribution
              sourceChangeInContribution<AggregationValueType, VertexValueType,
                                         GlobalInfoType>(
                  u, source_change_in_contribution[u],
                  vertexValueIdentity<VertexValueType>(),
                  vertex_values[iter - 1][u], global_info);
            }
          }
        }

        parallel_for(uintV u = 0; u < n; u++) {
          if (frontier_curr[u]) {
            // check for propagate and retract for the vertices.
            intE outDegree = my_graph.V[u].getOutDegree();
            granular_for(j, 0, outDegree, (outDegree > 1024), {
              uintV v = my_graph.V[u].getOutNeighbor(j);
              AggregationValueType contrib_change =
                  use_source_contribution
                      ? source_change_in_contribution[u]
                      : aggregationValueIdentity<AggregationValueType>();
#ifdef EDGEDATA
              EdgeData *edge_data = my_graph.V[u].getOutEdgeData(j);
#else
              EdgeData *edge_data = &emptyEdgeData;
#endif
              bool ret =
                  edgeFunction(u, v, *edge_data, vertex_values[iter - 1][u],
                               contrib_change, global_info);
              if (ret) {
                if (use_lock) {
#ifdef CONFIG_PROFILE_POINTS
                  PROFILE_START(PP_MUTEX_ACQ)
#endif
                  vertex_locks[v].writeLock();
#ifdef CONFIG_PROFILE_POINTS
                  PROFILE_LEAVE(PP_MUTEX_ACQ)
#endif

#ifdef CONFIG_PROFILE_POINTS
                  PROFILE_START(PP_MUTEX_CS)
#endif
                  addToAggregation(contrib_change, delta[v], global_info);
#ifdef CONFIG_PROFILE_POINTS
                  PROFILE_LEAVE(PP_MUTEX_CS)
#endif

#ifdef CONFIG_PROFILE_POINTS
                  PROFILE_START(PP_MUTEX_RLX)
#endif
                  vertex_locks[v].unlock();
#ifdef CONFIG_PROFILE_POINTS
                  PROFILE_LEAVE(PP_MUTEX_RLX)
#endif
                } else {
                  addToAggregationAtomic(contrib_change, delta[v], global_info);
                }
                if (!frontier_next[v])
                  frontier_next[v] = 1;
              }
            });
          }
        }

        phase_time = phase_timer.next();
        adaptive_executor.updateEdgeMapTime(iter, phase_time);

        // ========== VERTEX COMPUTATION ==========
        parallel_for(uintV v = 0; v < n; v++) {
          // Reset frontier for next iteration
          frontier_curr[v] = 0;
          // Process all vertices affected by EdgeMap
          if (frontier_next[v] ||
              forceComputeVertexForIteration(v, iter, global_info)) {

            frontier_next[v] = 0;
            // Update aggregation value and reset change received[v] (i.e.
            // delta[v])
            addToAggregation(delta[v], aggregation_values[iter][v],
                             global_info);
            delta[v] = aggregationValueIdentity<AggregationValueType>();

            // Calculate new_value based on the updated aggregation value
            VertexValueType new_value;
            computeFunction(v, aggregation_values[iter][v],
                            vertex_values[iter - 1][v], new_value, global_info);

            // Check if change is significant
            if (notDelZero(new_value, vertex_values[iter - 1][v], global_info)) {
              // change is significant. Update vertex_values
              vertex_values[iter][v] = new_value;
              // Set active for next iteration.
              frontier_curr[v] = 1;
            } else {
              // change is not significant. Copy vertex_values[iter-1]
              vertex_values[iter][v] = vertex_values[iter - 1][v];
            }
          }
          frontier_curr[v] =
              frontier_curr[v] ||
              forceActivateVertexForIteration(v, iter + 1, global_info);
          if (frontier_curr[v]) {
            if (use_source_contribution) {
              // update source_contrib for next iteration
              sourceChangeInContribution<AggregationValueType, VertexValueType,
                                         GlobalInfoType>(
                  v, source_change_in_contribution[v],
                  vertex_values[iter - 1][v], vertex_values[iter][v],
                  global_info);
            } else {
              source_change_in_contribution[v] =
                  aggregationValueIdentity<AggregationValueType>();
            }
          }
        }
        phase_time = phase_timer.stop();
        adaptive_executor.updateVertexMapTime(iter, phase_time);

        vertexSubset temp_vs(n, frontier_curr);
        frontier_curr_vs = temp_vs;
        misc_time += phase_timer.next();
        iteration_time = iteration_timer.stop();
        cout << "iteration:" << iter << " time:"  << iteration_time << "s" << endl;

        if (ae_enabled && iter == 1) {
          adaptive_executor.setApproximateTimeForCurrIter(iteration_time);
        }

#ifdef CONFIG_PROFILE_POINTS
        // MIND_TODO sleep for kernel profile points cleaning
        if (iter <= MIND_WARMUP_ITER) {
          atomic_int barrier;
          barrier.store(0, memory_order_release);
          int nWorkers = __cilkrts_get_nworkers();
          parallel_for_1(int t = 0; t < nWorkers; ++t) {
            clear_local_profile_points();
            printf("local profile point cleared by pthread[%ld] tid[%d], done warm up, sleep 60s, please clear kernel profile points\n", pthread_self(), t);
            sleep(60);
            barrier.fetch_add(1, memory_order_release);
            while (barrier.load(memory_order_acquire) != nWorkers)
              ;
          }
        }
        // MIND_TODO report thread local profile
        if (iter == max_iterations - 1) {
          int nWorkers = __cilkrts_get_nworkers();
          parallel_for_1(int t = 0; t < nWorkers; ++t) {
            report_local_profile_points();
            printf("local profile point reported by pthread[%ld] tid[%d]\n", pthread_self(), t);
            sleep(1);
          }
        }
#endif
        // Convergence check
        converged_iteration = iter;
        if (frontier_curr_vs.isEmpty()) {
          break;
        }
      }
    }
    if (ae_enabled) {
      adaptive_executor.updateEquation(converged_iteration);
    }

#ifdef CONFIG_PROFILE_POINTS
    print_global_profile_points();
#endif

    //MIND_TODO
    double mind_time = mind_timer.stop();
    cout << "mind time:"  << mind_time << "s" << endl;
    parallel_for_1(int t = 0; t < 100000; ++t) {
      // brk(0);
      sleep(600);
    }

    return converged_iteration;
  }

  // ======================================================================
  // DELTACOMPUTE
  // ======================================================================
  void deltaCompute(edgeArray &edge_additions, edgeArray &edge_deletions) {
    timer iteration_timer, phase_timer, full_timer, pre_compute_timer;
    double misc_time, copy_time, phase_time, iteration_time, pre_compute_time;
    iteration_time = 0;
    full_timer.start();

    // TODO : Realloc addition of new vertices
    n_old = n;
    if (edge_additions.maxVertex >= n) {
      processVertexAddition(edge_additions.maxVertex);
    }

    // Reset values before incremental computation
    parallel_for(uintV v = 0; v < n; v++) {
      frontier_curr[v] = 0;
      frontier_next[v] = 0;
      changed[v] = 0;

      vertex_value_old_prev[v] = vertexValueIdentity<VertexValueType>();
      vertex_value_old_curr[v] = vertexValueIdentity<VertexValueType>();
      initializeVertexValue<VertexValueType>(v, vertex_value_old_next[v],
                                             global_info);

      delta[v] = aggregationValueIdentity<AggregationValueType>();
      if (use_source_contribution) {
        source_change_in_contribution[v] =
            aggregationValueIdentity<AggregationValueType>();
      }
    }

    // ==================== UPDATE GLOBALINFO ===============================
    // deltaCompute/initCompute Save a copy of global_info before we lose any
    // relevant information of the old graph For example, In PageRank, we need
    // to save the outDegree for all vertices corresponding to the old graph
    global_info_old.copy(global_info);

    // Update global_info based on edge additions or deletions. This is
    // application specific. For example, for pagerank, the the outDegree of
    // vertices with edgeAddition will increase and those with edgeDeletions
    // will decrease
    global_info.processUpdates(edge_additions, edge_deletions);

    // ========== EDGE COMPUTATION - DIRECT CHANGES - for first iter ==========
    pre_compute_timer.start();
    parallel_for(long i = 0; i < edge_additions.size; i++) {
      uintV source = edge_additions.E[i].source;
      uintV destination = edge_additions.E[i].destination;

      // Update frontier and changed values
      hasSourceChangedByUpdate(source, edge_addition_enum,
                               frontier_curr[source], changed[source],
                               global_info, global_info_old);
      hasSourceChangedByUpdate(destination, edge_addition_enum,
                               frontier_curr[destination], changed[destination],
                               global_info, global_info_old);
      if (forceActivateVertexForIteration(source, 1, global_info_old)) {

        if (frontier_curr[source]) {
          changed[source] = true;
        }
        if (frontier_curr[destination]) {
          changed[source] = true;
        }

        AggregationValueType contrib_change;
        if (use_source_contribution) {
          sourceChangeInContribution<AggregationValueType, VertexValueType,
                                     GlobalInfoType>(
              source, contrib_change, vertexValueIdentity<VertexValueType>(),
              vertex_values[0][source], global_info_old);
        }

// Do repropagate for edge source->destination.
#ifdef EDGEDATA
        EdgeData *edge_data = edge_additions.E[i].edgeData;
#else
        EdgeData *edge_data = &emptyEdgeData;
#endif
        bool ret =
            edgeFunction(source, destination, *edge_data,
                         vertex_values[0][source], contrib_change, global_info);
        if (ret) {
          if (use_lock) {
            vertex_locks[destination].writeLock();
            addToAggregation(contrib_change, delta[destination],
                             global_info_old);
            vertex_locks[destination].unlock();
          } else {
            addToAggregationAtomic(contrib_change, delta[destination],
                                   global_info_old);
          }
          if (!changed[destination])
            changed[destination] = true;
        }
      }
    }

    parallel_for(long i = 0; i < edge_deletions.size; i++) {
      uintV source = edge_deletions.E[i].source;
      uintV destination = edge_deletions.E[i].destination;

      hasSourceChangedByUpdate(source, edge_deletion_enum,
                               frontier_curr[source], changed[source],
                               global_info, global_info_old);
      hasSourceChangedByUpdate(destination, edge_deletion_enum,
                               frontier_curr[destination], changed[destination],
                               global_info, global_info_old);
      if (forceActivateVertexForIteration(source, 1, global_info_old)) {
        // Update frontier and changed values
        if (frontier_curr[source]) {
          changed[source] = true;
        }
        if (frontier_curr[destination]) {
          changed[source] = true;
        }

        AggregationValueType contrib_change;
        if (use_source_contribution) {
          sourceChangeInContribution<AggregationValueType, VertexValueType,
                                     GlobalInfoType>(
              source, contrib_change, vertexValueIdentity<VertexValueType>(),
              vertex_values[0][source], global_info_old);
        }

// Do retract for edge source->destination
#ifdef EDGEDATA
        EdgeData *edge_data = edge_deletions.E[i].edgeData;
#else
        EdgeData *edge_data = &emptyEdgeData;
#endif
        bool ret = edgeFunction(source, destination, *edge_data,
                                vertex_values[0][source], contrib_change,
                                global_info_old);
        if (ret) {
          if (use_lock) {
            vertex_locks[destination].writeLock();
            removeFromAggregation(contrib_change, delta[destination],
                                  global_info_old);
            vertex_locks[destination].unlock();
          } else {
            removeFromAggregationAtomic(contrib_change, delta[destination],
                                        global_info_old);
          }
          if (!changed[destination])
            changed[destination] = true;
        }
      }
    }
    pre_compute_time = pre_compute_timer.stop();

    // =============== INCREMENTAL COMPUTE - REFINEMENT START ================
    vertexSubset frontier_curr_vs(n, frontier_curr);
    bool should_switch_now = false;
    bool use_delta = true;

    if (ae_enabled && shouldSwitch(0, 0)) {
      should_switch_now = true;
    }

    for (int iter = 1; iter < max_iterations; iter++) {
      // Perform switch if needed
      if (should_switch_now) {
        converged_iteration = performSwitch(iter);
        break;
      }

      // initialize timers
      {
        iteration_timer.start();
        phase_timer.start();
        iteration_time = 0;
        misc_time = 0;
        copy_time = 0;
      }
      use_delta = shouldUseDelta(iter);

      // ================ COPY - PREPARE CURRENT ITERATION ================
      {
        VertexValueType *temp1 = vertex_value_old_prev;
        vertex_value_old_prev = vertex_value_old_curr;
        vertex_value_old_curr = vertex_value_old_next;
        vertex_value_old_next = temp1;

        if (iter <= converged_iteration) {
          parallel_for(uintV v = 0; v < n; v++) {
            vertex_value_old_next[v] = vertex_values[iter][v];
          }
        } else {
          converged_iteration = performSwitch(iter);
          break;
        }
      }
      copy_time += phase_timer.next();
      // ========== EDGE COMPUTATION - TRANSITIVE CHANGES ==========
      if ((use_source_contribution) && (iter == 1)) {
        // Compute source contribution for first iteration
        parallel_for(uintV u = 0; u < n; u++) {
          if (frontier_curr[u]) {
            // compute source change in contribution
            AggregationValueType contrib_change =
                aggregationValueIdentity<AggregationValueType>();
            sourceChangeInContribution<AggregationValueType, VertexValueType,
                                       GlobalInfoType>(
                u, contrib_change, vertexValueIdentity<VertexValueType>(),
                vertex_values[iter - 1][u], global_info);
            addToAggregation(contrib_change, source_change_in_contribution[u],
                             global_info);
            sourceChangeInContribution<AggregationValueType, VertexValueType,
                                       GlobalInfoType>(
                u, contrib_change, vertexValueIdentity<VertexValueType>(),
                vertex_value_old_curr[u], global_info_old);
            removeFromAggregation(
                contrib_change, source_change_in_contribution[u], global_info);
          }
        }
      }

      parallel_for(uintV u = 0; u < n; u++) {
        if (frontier_curr[u]) {
          // check for propagate and retract for the vertices.
          intE outDegree = my_graph.V[u].getOutDegree();

          granular_for(i, 0, outDegree, (outDegree > 1024), {
            uintV v = my_graph.V[u].getOutNeighbor(i);
            bool ret = false;
            AggregationValueType contrib_change =
                use_source_contribution
                    ? source_change_in_contribution[u]
                    : aggregationValueIdentity<AggregationValueType>();

#ifdef EDGEDATA
            EdgeData *edge_data = my_graph.V[u].getOutEdgeData(i);
#else
            EdgeData *edge_data = &emptyEdgeData;
#endif
            ret = edgeFunction(u, v, *edge_data, vertex_values[iter - 1][u],
                               contrib_change, global_info);

            if (ret) {
              if (use_lock) {
                vertex_locks[v].writeLock();
                if (ret) {
                  addToAggregation(contrib_change, delta[v], global_info);
                }
                vertex_locks[v].unlock();

              } else {
                if (ret) {
                  addToAggregationAtomic(contrib_change, delta[v], global_info);
                }
              }
              if (!changed[v])
                changed[v] = 1;
            }
          });
        }
      }
      phase_time = phase_timer.next();

      // ========== VERTEX COMPUTATION  ==========
      bool use_delta_next_iteration = shouldUseDelta(iter + 1);
      parallel_for(uintV v = 0; v < n; v++) {
        // changed vertices need to be processed
        frontier_curr[v] = 0;
        if ((v >= n_old) && (changed[v] == false)) {
          changed[v] = forceComputeVertexForIteration(v, iter, global_info);
        }

        if (changed[v]) {
          frontier_curr[v] = 0;

          // delta has the current cumulative change for the vertex.
          // Update the aggregation value in history
          addToAggregation(delta[v], aggregation_values[iter][v], global_info);

          VertexValueType new_value;
          computeFunction(v, aggregation_values[iter][v],
                          vertex_values[iter - 1][v], new_value, global_info);

          if (forceActivateVertexForIteration(v, iter + 1, global_info)) {
            frontier_curr[v] = 1;
          }
          AggregationValueType contrib_change =
              aggregationValueIdentity<AggregationValueType>();
          source_change_in_contribution[v] =
              aggregationValueIdentity<AggregationValueType>();

          if (notDelZero(new_value, vertex_values[iter - 1][v], global_info)) {
            // change is significant. Update vertex_values
            vertex_values[iter][v] = new_value;
            frontier_curr[v] = 1;
            if (use_delta_next_iteration) {
              sourceChangeInContribution<AggregationValueType, VertexValueType,
                                         GlobalInfoType>(
                  v, contrib_change, vertex_values[iter - 1][v],
                  vertex_values[iter][v], global_info);
            } else {
              sourceChangeInContribution<AggregationValueType, VertexValueType,
                                         GlobalInfoType>(
                  v, contrib_change, vertexValueIdentity<VertexValueType>(),
                  vertex_values[iter][v], global_info);
            }
            addToAggregation(contrib_change, source_change_in_contribution[v],
                             global_info);


          } else {
            // change is not significant. Copy vertex_values[iter-1]
            vertex_values[iter][v] = vertex_values[iter - 1][v];
          }

          if (notDelZero(vertex_value_old_next[v], vertex_value_old_curr[v],
                        global_info_old)) {
            // change is significant. Update v_change
            frontier_curr[v] = 1;
            if (use_delta_next_iteration) {
              sourceChangeInContribution<AggregationValueType, VertexValueType,
                                         GlobalInfoType>(
                  v, contrib_change, vertex_value_old_curr[v],
                  vertex_value_old_next[v], global_info_old);
            } else {

              sourceChangeInContribution<AggregationValueType, VertexValueType,
                                         GlobalInfoType>(
                  v, contrib_change, vertexValueIdentity<VertexValueType>(),
                  vertex_value_old_next[v], global_info_old);
            }
            removeFromAggregation(contrib_change,
                                  source_change_in_contribution[v],
                                  global_info_old);
          }
        }
      }
      phase_time = phase_timer.next();

      // ========== EDGE COMPUTATION - DIRECT CHANGES - for next iter ==========
      bool has_direct_changes = false;
      parallel_for(long i = 0; i < edge_additions.size; i++) {
        uintV source = edge_additions.E[i].source;
        uintV destination = edge_additions.E[i].destination;
        AggregationValueType contrib_change;

        if (notDelZero(vertex_value_old_curr[source],
                      vertex_value_old_next[source], global_info_old) ||
            (forceActivateVertexForIteration(source, iter + 1,
                                             global_info_old))) {
          if (use_delta_next_iteration) {
            sourceChangeInContribution<AggregationValueType, VertexValueType,
                                       GlobalInfoType>(
                source, contrib_change, vertex_value_old_curr[source],
                vertex_value_old_next[source], global_info_old);
          } else {
            sourceChangeInContribution<AggregationValueType, VertexValueType,
                                       GlobalInfoType>(
                source, contrib_change, vertexValueIdentity<VertexValueType>(),
                vertex_value_old_next[source], global_info_old);
          }
// Do repropagate for edge source->destination.
#ifdef EDGEDATA
          EdgeData *edge_data = edge_additions.E[i].edgeData;
#else
          EdgeData *edge_data = &emptyEdgeData;
#endif
          bool ret = edgeFunction(source, destination, *edge_data,
                                  vertex_values[0][source], contrib_change,
                                  global_info);

          if (ret) {
            if (use_lock) {
              vertex_locks[destination].writeLock();
              addToAggregation(contrib_change, delta[destination],
                               global_info_old);
              vertex_locks[destination].unlock();
            } else {
              addToAggregationAtomic(contrib_change, delta[destination],
                                     global_info_old);
            }
            if (!changed[destination])
              changed[destination] = 1;
            if (!has_direct_changes)
              has_direct_changes = true;
          }
        }
      }

      parallel_for(long i = 0; i < edge_deletions.size; i++) {
        uintV source = edge_deletions.E[i].source;
        uintV destination = edge_deletions.E[i].destination;
        AggregationValueType contrib_change;

        if (notDelZero(vertex_value_old_curr[source],
                      vertex_value_old_next[source], global_info_old) ||
            (forceActivateVertexForIteration(source, iter + 1,
                                             global_info_old))) {
          // Do repropagate for edge source->destination.
          if (use_delta_next_iteration) {
            sourceChangeInContribution<AggregationValueType, VertexValueType,
                                       GlobalInfoType>(
                source, contrib_change, vertex_value_old_curr[source],
                vertex_value_old_next[source], global_info_old);
          } else {
            sourceChangeInContribution<AggregationValueType, VertexValueType,
                                       GlobalInfoType>(
                source, contrib_change, vertexValueIdentity<VertexValueType>(),
                vertex_value_old_next[source], global_info_old);
          }
#ifdef EDGEDATA
          EdgeData *edge_data = edge_deletions.E[i].edgeData;
#else
          EdgeData *edge_data = &emptyEdgeData;
#endif
          bool ret = edgeFunction(source, destination, *edge_data,
                                  vertex_values[0][source], contrib_change,
                                  global_info);

          if (ret) {
            if (use_lock) {
              vertex_locks[destination].writeLock();
              removeFromAggregation(contrib_change, delta[destination],
                                    global_info_old);
              vertex_locks[destination].unlock();

            } else {
              removeFromAggregationAtomic(contrib_change, delta[destination],
                                          global_info_old);
            }
            if (!changed[destination])
              changed[destination] = 1;
            if (!has_direct_changes)
              has_direct_changes = true;
          }
        }
      }
      phase_time = phase_timer.next();

      vertexSubset temp_vs(n, frontier_curr);
      frontier_curr_vs = temp_vs;

      misc_time += phase_timer.next();
      iteration_time = iteration_timer.next();

      // Convergence check
      if (!has_direct_changes && frontier_curr_vs.isEmpty()) {
        // There are no more active vertices
        if (iter == converged_iteration) {
          break;
        } else if (iter > converged_iteration) {
          assert(("Missed switching to Traditional incremental computing when "
                  "iter == converged_iter",
                  false));
        } else {
          // Values stable for the changed vertices at this iteration.
          // But, the changed vertices might receive new changes. So,
          // continue loop until iter == converged_iteration vertices may
          // still not have converged. So, keep continuing until
          // converged_iteration is reached.
        }
      }
      if (iter == 1) {
        iteration_time += pre_compute_time;
      }

      if (ae_enabled && shouldSwitch(iter, iteration_time)) {
        should_switch_now = true;
      }
      misc_time += phase_timer.stop();
      iteration_time += iteration_timer.stop();
    }

    cout << "Finished batch : " << full_timer.stop() << "\n";
    cout << "Number of iterations : " << converged_iteration << "\n";
    // testPrint();
    printOutput();
  }

  // Refactor this in a better way
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::my_graph;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::config;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::max_iterations;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::history_iterations;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::converged_iteration;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::use_lock;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::vertex_locks;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::aggregation_values;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::vertex_values;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::n;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::global_info;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::delta;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::use_source_contribution;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::source_change_in_contribution;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::n_old;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::global_info_old;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::vertex_value_old_next;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::vertex_value_old_curr;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::vertex_value_old_prev;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::all;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::frontier_curr;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::frontier_next;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::changed;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::ingestor;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::current_batch;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::adaptive_executor;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::ae_enabled;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::testPrint;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::printOutput;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::shouldSwitch;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::performSwitch;
  using GraphBoltEngine<vertex, AggregationValueType, VertexValueType,
                        GlobalInfoType>::processVertexAddition;
};
#endif
