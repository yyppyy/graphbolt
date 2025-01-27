sudo apt-get install -y libcilkrts5

#succ command
/usr/bin/g++-7 -g -O3 -DCILK  -DEDGELONG  PageRank.C  -o PageRank -fcilkplus -lcilkrts -ldl -static -std=c++14


#sample run
./PageRank -nWorkers 4 -numberOfUpdateBatches 2 -nEdges 5 -streamPath ../inputs/sample_edge_operations.empty.txt ../inputs/sample_graph.adj 


#static graph
./PageRank -nWorkers 4 -numberOfUpdateBatches 1 -nEdges 0 -streamPath ../inputs/sample_edge_operations.empty.txt ../inputs/sample_graph.adj


# wiki graph 10+GB mem
./PageRank -nWorkers 8 -maxIters 1 -numberOfUpdateBatches 1 -nEdges 0 -streamPath ../inputs/sample_edge_operations.empty.txt /media/data_ssds/yanpeng/wikipedia_link_en/wikipedia_link_en.adj


# youtube link graph 1.2GB mem
./PageRank -nWorkers 40 -maxIters 10 -numberOfUpdateBatches 1 -nEdges 0 -streamPath ../inputs/sample_edge_operations.empty.txt /media/data_ssds/yanpeng/youtube-links/youtube-links.adj

# youtube friendship graph
./PageRank -nWorkers 8 -maxIters 5 -numberOfUpdateBatches 1 -nEdges 0 -streamPath ../inputs/sample_edge_operations.empty.txt /media/data_ssds/yanpeng/com-youtube/com-youtube.adj