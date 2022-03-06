sudo apt-get install -y libcilkrts5

#succ command
/usr/bin/g++-7 -g -O3 -DCILK  -DEDGELONG  PageRank.C  -o PageRank -fcilkplus -lcilkrts -ldl -static -std=c++14


#sample run
./PageRank -nWorkers 4 -numberOfUpdateBatches 2 -nEdges 5 -streamPath ../inputs/sample_edge_operations.empty.txt ../inputs/sample_graph.adj 


#static graph
./PageRank -nWorkers 4 -numberOfUpdateBatches 1 -nEdges 0 -streamPath ../inputs/sample_edge_operations.empty.txt ../inputs/sample_graph.adj