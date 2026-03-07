# Linear Algebra

## Matrices
Multiplication: For $A_{m \times n}$ and $B_{n \times p}$, the product $AB$ is of size $m \times p$.
Transpose: $(AB)^T = B^T A^T$
Inverse: $A A^{-1} = I $. A $2 \times 2$ matrix $A = \begin{bmatrix}a & b \\ c & d\end{bmatrix}$ has an inverse if its determinant is non-zero:
$A^{-1} = \frac{1}{ad-bc} \begin{bmatrix}d & -b \\ -c & a\end{bmatrix}$

## Determinants
For a $2 \times 2$ matrix:
$$\det(A) = ad - bc$$
For a $3 \times 3$ matrix:
$$\det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)$$
Properties:
- $\det(AB) = \det(A)\det(B)$
- $\det(A^T) = \det(A)$
- If two rows (or columns) are identical, the determinant is 0.

## Vectors
Dot Product: $\mathbf{u} \cdot \mathbf{v} = |\mathbf{u}||\mathbf{v}|\cos(\theta)$
Cross Product: Magnitude is $|\mathbf{u} \times \mathbf{v}| = |\mathbf{u}||\mathbf{v}|\sin(\theta)$. The direction is perpendicular to both $\mathbf{u}$ and $\mathbf{v}$.
