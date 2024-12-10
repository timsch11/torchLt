Home of every line of code that is immediately related to cuda, the files are distinguished by different abstractional levels:
- cudaMath includes very basic linear algebra operations (i.e matrix multiplication)
- cudaMem handles memory initialization
- util provides useful tools for error checking or to calculate block/thread allocation
- cudaDif handles the execution of operations related to differentiation (cudaMath dependency)
- cudaNN uses all of the above mentioned files to provide an interface like abstraction to call lower level cuda operations

- Tensor offers common operations applied to tensors (or related to neural networks), mostly calls cudaNN and cudaDif; features automatic differentiation