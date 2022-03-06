sudo apt-get install -y libcilkrts5

#succ command
/usr/bin/g++-7 -g -O3 -DCILK  -DEDGELONG  PageRank.C  -o PageRank -fcilkplus -lcilkrts -ldl -static -std=c++14


#sample run
./PageRank -numberOfUpdateBatches 2 -nEdges 5 -streamPath ../inputs/sample_edge_operations.txt -outputFile pr_output ../inputs/sample_graph.adj 