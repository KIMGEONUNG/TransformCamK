## Test Definition

Let $\mathcal{R}$ be a rendering function, $M$ be a 3D model, $K$ is an camera intrinsic parameter, $\mathcal{T}_I$ be a image transformation function, $\mathcal{T}_K$ be a camera intrinsic transformation function. Our goal is to implement the function $\mathcal{T}_K$. In formal terms, we have:

$$
\mathcal{T}_I(\mathcal{R}(M, K)) = \mathcal{R}(M, \mathcal{T}_K(K, \mathcal{T}_I)).
$$

For testing purposes, we require multiple pairs of $\mathcal{T}_I$ and $K$.
