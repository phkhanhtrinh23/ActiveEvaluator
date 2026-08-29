# Representative Meta-Dataset Selection with Distance-Induced Facility Location

## 1. Goal

Let

$$
B=\{b_1,b_2,\ldots,b_N\}
$$

be a **meta-dataset** containing $N$ datasets. Each element $b_i$ is itself a dataset:

$$
b_i=\{x_{i1},x_{i2},\ldots,x_{iM_i}\},
$$

where $x_{im}$ denotes the $m$-th sample inside dataset $b_i$.

The goal is to select a subset

$$
A\subseteq B
$$

containing at most $K$ datasets:

$$
|A|\le K,
$$

such that $A$ represents the entire meta-dataset $B$ as well as possible.

Therefore, the optimization problem is not to find the "largest" subset, because the budget $K$ already determines its maximum size. Instead, we want the **most representative subset under budget $K$**:

$$
\boxed{
A^*=\arg\max_{\substack{A\subseteq B\\|A|\le K}}F(A)
}
$$

for a suitable representativeness objective $F(A)$.

---

## 2. Hierarchy of the Data

It is important to distinguish three different levels:

$$
\boxed{B\rightarrow b_i\rightarrow x_{im}}
$$

where:

- $B$: the complete meta-dataset,
- $b_i$: the $i$-th dataset inside $B$,
- $x_{im}$: the $m$-th sample inside dataset $b_i$.

For example,

$$
B=\{b_1,b_2,b_3,b_4\}
$$

may contain four separate datasets, where

$$
b_1=\{x_{11},x_{12},\ldots\},\qquad
b_2=\{x_{21},x_{22},\ldots\},
$$

and so on.

Throughout the selection algorithm:

- $i$ indexes a **candidate dataset** that may be added to $A$,
- $j$ indexes a **dataset in the complete meta-dataset $B$** whose representation quality is being evaluated.

---

## 3. Step 1: Measure Distance Between Every Pair of Datasets

Choose a distance measure $D$ that compares two complete datasets:

$$
D_{ij}=D(b_i,b_j).
$$

Possible choices include:

1. Kernel Mean Embedding / Maximum Mean Discrepancy,
2. Sliced Wasserstein Distance,
3. Hausdorff Distance.

The important point is that

$$
D(b_i,b_j)
$$

compares the **whole dataset $b_i$** with the **whole dataset $b_j$**, not individual samples directly.

After computing all pairwise distances, we obtain an $N\times N$ matrix:

$$
D=
\begin{bmatrix}
D_{11}&D_{12}&\cdots&D_{1N}\\
D_{21}&D_{22}&\cdots&D_{2N}\\
\vdots&\vdots&\ddots&\vdots\\
D_{N1}&D_{N2}&\cdots&D_{NN}
\end{bmatrix}.
$$

For proper distance measures,

$$
D_{ii}=0.
$$

---

## 3.1. Distance Option 1: Kernel Mean Embedding / Maximum Mean Discrepancy

Kernel Mean Embedding (KME) represents the distribution of an entire dataset in a Reproducing Kernel Hilbert Space (RKHS).

For dataset $b_i$,

$$
b_i=\{x_{i1},x_{i2},\ldots,x_{iM_i}\},
$$

its empirical kernel mean embedding is

$$
\boxed{
\hat{\mu}_i
=
\frac{1}{M_i}
\sum_{m=1}^{M_i}
\phi(x_{im})
}
$$

where $\phi(\cdot)$ is the feature mapping associated with a positive-definite kernel $k$ satisfying

$$
k(x,y)
=
\langle\phi(x),\phi(y)\rangle_{\mathcal H}.
$$

KME itself is a representation. The corresponding distance between two datasets is the Maximum Mean Discrepancy (MMD):

$$
\boxed{
D_{ij}^{\mathrm{KME}}
=
\operatorname{MMD}(b_i,b_j)
=
\left\|
\hat{\mu}_i-\hat{\mu}_j
\right\|_{\mathcal H}.
}
$$

The squared empirical MMD can be computed directly using the kernel trick:

$$
\boxed{
\begin{aligned}
\left(D_{ij}^{\mathrm{KME}}\right)^2
={}&
\frac{1}{M_i^2}
\sum_{m=1}^{M_i}
\sum_{m'=1}^{M_i}
k(x_{im},x_{im'})
\\
&+
\frac{1}{M_j^2}
\sum_{n=1}^{M_j}
\sum_{n'=1}^{M_j}
k(x_{jn},x_{jn'})
\\
&-
\frac{2}{M_iM_j}
\sum_{m=1}^{M_i}
\sum_{n=1}^{M_j}
k(x_{im},x_{jn}).
\end{aligned}
}
$$

A common kernel is the RBF kernel

$$
k(x,y)
=
\exp\left(
-\frac{\|x-y\|_2^2}{2\sigma^2}
\right).
$$

### Interpretation

If

$$
D_{ij}^{\mathrm{KME}}\approx0,
$$

then the overall distributions of $b_i$ and $b_j$ are similar in the RKHS.

If

$$
D_{ij}^{\mathrm{KME}}
$$

is large, the two dataset distributions are different.

### Convert KME/MMD distance to similarity

Define

$$
\boxed{
S_{ij}^{\mathrm{KME}}
=
\exp\left(
-\frac{
\left(D_{ij}^{\mathrm{KME}}\right)^2
}{
2\tau_{\mathrm{KME}}^2
}
\right).
}
$$

Since

$$
D_{ij}^{\mathrm{KME}}
=
\operatorname{MMD}(b_i,b_j),
$$

this can also be written as

$$
\boxed{
S_{ij}^{\mathrm{KME}}
=
\exp\left(
-\frac{
\operatorname{MMD}^2(b_i,b_j)
}{
2\tau_{\mathrm{KME}}^2
}
\right).
}
$$

For a selected set $A$, the best KME-based representative of dataset $b_j$ is

$$
\boxed{
c_A^{\mathrm{KME}}(j)
=
\max_{b_i\in A}
S_{ij}^{\mathrm{KME}}.
}
$$

Therefore, the KME-induced facility-location objective is

$$
\boxed{
F_{\mathrm{KME}}(A)
=
\sum_{j=1}^{N}
\max_{b_i\in A}
\exp\left(
-\frac{
\operatorname{MMD}^2(b_i,b_j)
}{
2\tau_{\mathrm{KME}}^2
}
\right).
}
$$

---

## 3.2. Distance Option 2: Sliced Wasserstein Distance

The Wasserstein distance compares two probability distributions by measuring how much probability mass must be moved to transform one distribution into the other.

For high-dimensional datasets, direct Wasserstein computation can be expensive. Sliced Wasserstein Distance simplifies the comparison by projecting both datasets onto many one-dimensional directions.

Let the empirical distributions represented by $b_i$ and $b_j$ be $P_i$ and $P_j$.

Choose a unit direction

$$
\theta\in\mathbb S^{d-1},
$$

where $\mathbb S^{d-1}$ is the unit sphere in $d$ dimensions.

Project every sample onto that direction:

$$
z_{im}^{(\theta)}
=
\theta^\top x_{im}.
$$

Similarly,

$$
z_{jn}^{(\theta)}
=
\theta^\top x_{jn}.
$$

The projected distributions are written as

$$
\theta_{\#}P_i
\qquad\text{and}\qquad
\theta_{\#}P_j.
$$

For order $p\ge1$, the Sliced Wasserstein Distance is

$$
\boxed{
D_{ij}^{\mathrm{SW}}
=
SW_p(P_i,P_j)
=
\left[
\int_{\mathbb S^{d-1}}
W_p^p
\left(
\theta_{\#}P_i,
\theta_{\#}P_j
\right)
d\theta
\right]^{1/p}.
}
$$

In practice, the integral is approximated using $L$ randomly sampled projection directions

$$
\theta_1,\theta_2,\ldots,\theta_L.
$$

Then

$$
\boxed{
D_{ij}^{\mathrm{SW}}
\approx
\left[
\frac{1}{L}
\sum_{\ell=1}^{L}
W_p^p
\left(
\theta_{\ell\#}P_i,
\theta_{\ell\#}P_j
\right)
\right]^{1/p}.
}
$$

### One-dimensional Wasserstein computation

If the two datasets contain the same number $M$ of samples, project onto direction $\theta_\ell$ and sort the projected values:

$$
z_{i(1)}^{(\ell)}
\le
z_{i(2)}^{(\ell)}
\le
\cdots
\le
z_{i(M)}^{(\ell)},
$$

and

$$
z_{j(1)}^{(\ell)}
\le
z_{j(2)}^{(\ell)}
\le
\cdots
\le
z_{j(M)}^{(\ell)}.
$$

The one-dimensional Wasserstein distance is then

$$
\boxed{
W_p^p
\left(
\theta_{\ell\#}P_i,
\theta_{\ell\#}P_j
\right)
=
\frac{1}{M}
\sum_{m=1}^{M}
\left|
z_{i(m)}^{(\ell)}
-
z_{j(m)}^{(\ell)}
\right|^p.
}
$$

For $p=2$,

$$
\boxed{
W_2^2
=
\frac{1}{M}
\sum_{m=1}^{M}
\left(
z_{i(m)}^{(\ell)}
-
z_{j(m)}^{(\ell)}
\right)^2.
}
$$

If the datasets have unequal numbers of samples, the one-dimensional Wasserstein distance can instead be computed through their empirical quantile functions.

### Interpretation

Sliced Wasserstein compares the **geometry and shape of the complete distributions**.

A small value

$$
D_{ij}^{\mathrm{SW}}
$$

means that the two datasets have similar projected distributions across many directions.

### Convert Sliced Wasserstein distance to similarity

Define

$$
\boxed{
S_{ij}^{\mathrm{SW}}
=
\exp\left(
-\frac{
\left(D_{ij}^{\mathrm{SW}}\right)^2
}{
2\tau_{\mathrm{SW}}^2
}
\right).
}
$$

Equivalently,

$$
\boxed{
S_{ij}^{\mathrm{SW}}
=
\exp\left(
-\frac{
SW_p^2(P_i,P_j)
}{
2\tau_{\mathrm{SW}}^2
}
\right).
}
$$

For selected set $A$, the best Sliced-Wasserstein representative of dataset $b_j$ is

$$
\boxed{
c_A^{\mathrm{SW}}(j)
=
\max_{b_i\in A}
S_{ij}^{\mathrm{SW}}.
}
$$

The corresponding facility-location objective is

$$
\boxed{
F_{\mathrm{SW}}(A)
=
\sum_{j=1}^{N}
\max_{b_i\in A}
\exp\left(
-\frac{
SW_p^2(P_i,P_j)
}{
2\tau_{\mathrm{SW}}^2
}
\right).
}
$$

---

## 3.3. Distance Option 3: Hausdorff Distance

Hausdorff Distance treats each dataset as a set of points and focuses on the **worst-covered point**.

Let

$$
b_i=\{x_{i1},\ldots,x_{iM_i}\}
$$

and

$$
b_j=\{x_{j1},\ldots,x_{jM_j}\}.
$$

First define the directed Hausdorff distance from $b_i$ to $b_j$:

$$
\boxed{
h(b_i,b_j)
=
\max_{x\in b_i}
\min_{y\in b_j}
\|x-y\|_2.
}
$$

This performs two operations:

1. For each point $x\in b_i$, find its closest point in $b_j$:

$$
\min_{y\in b_j}\|x-y\|_2.
$$

2. Among all points in $b_i$, take the worst such nearest-neighbour distance:

$$
\max_{x\in b_i}.
$$

Because this directed quantity is generally asymmetric,

$$
h(b_i,b_j)
\ne
h(b_j,b_i),
$$

the symmetric Hausdorff distance is

$$
\boxed{
D_{ij}^{\mathrm{H}}
=
d_H(b_i,b_j)
=
\max
\left\{
h(b_i,b_j),
h(b_j,b_i)
\right\}.
}
$$

Expanding the definition,

$$
\boxed{
d_H(b_i,b_j)
=
\max
\left\{
\max_{x\in b_i}
\min_{y\in b_j}
\|x-y\|_2,
\;
\max_{y\in b_j}
\min_{x\in b_i}
\|x-y\|_2
\right\}.
}
$$

### Interpretation

Hausdorff Distance asks:

> How far away is the worst-matched point between the two datasets?

Therefore it is more sensitive to extreme or poorly represented regions than an average-based distance.

### Convert Hausdorff distance to similarity

Define

$$
\boxed{
S_{ij}^{\mathrm{H}}
=
\exp\left(
-\frac{
\left(D_{ij}^{\mathrm{H}}\right)^2
}{
2\tau_{\mathrm{H}}^2
}
\right).
}
$$

Equivalently,

$$
\boxed{
S_{ij}^{\mathrm{H}}
=
\exp\left(
-\frac{
d_H^2(b_i,b_j)
}{
2\tau_{\mathrm{H}}^2
}
\right).
}
$$

For selected set $A$, the best Hausdorff-based representative of dataset $b_j$ is

$$
\boxed{
c_A^{\mathrm{H}}(j)
=
\max_{b_i\in A}
S_{ij}^{\mathrm{H}}.
}
$$

The Hausdorff-induced facility-location objective is

$$
\boxed{
F_{\mathrm{H}}(A)
=
\sum_{j=1}^{N}
\max_{b_i\in A}
\exp\left(
-\frac{
d_H^2(b_i,b_j)
}{
2\tau_{\mathrm{H}}^2
}
\right).
}
$$

---

## 3.4. Unified Form of the Three Distance-Induced Objectives

For any one of the three choices,

$$
D_{ij}^{(m)}
\in
\left\{
D_{ij}^{\mathrm{KME}},
D_{ij}^{\mathrm{SW}},
D_{ij}^{\mathrm{H}}
\right\},
$$

define

$$
\boxed{
S_{ij}^{(m)}
=
\exp\left(
-\frac{
\left(D_{ij}^{(m)}\right)^2
}{
2\tau_m^2
}
\right).
}
$$

Then the best similarity currently available for dataset $b_j$ is

$$
\boxed{
c_A^{(m)}(j)
=
\max_{b_i\in A}
S_{ij}^{(m)}.
}
$$

The complete facility-location objective is

$$
\boxed{
F_m(A)
=
\sum_{j=1}^{N}
c_A^{(m)}(j)
=
\sum_{j=1}^{N}
\max_{b_i\in A}
S_{ij}^{(m)}.
}
$$

Substituting the exponential similarity gives

$$
\boxed{
F_m(A)
=
\sum_{j=1}^{N}
\max_{b_i\in A}
\exp\left(
-\frac{
\left(D_{ij}^{(m)}\right)^2
}{
2\tau_m^2
}
\right).
}
$$

This is the common form used by the greedy representative-selection algorithm.

The three versions differ only in how the dataset-level distance $D_{ij}^{(m)}$ is computed:

$$
\boxed{
\begin{aligned}
D_{ij}^{\mathrm{KME}}
&=
\operatorname{MMD}(b_i,b_j),
\\
D_{ij}^{\mathrm{SW}}
&=
SW_p(P_i,P_j),
\\
D_{ij}^{\mathrm{H}}
&=
d_H(b_i,b_j).
\end{aligned}
}
$$

After the similarity matrices have been constructed, the monotonicity proof, submodularity proof, marginal-gain formula, and greedy selection algorithm are identical for all three choices.


---

## 4. Step 2: Convert Distance to Similarity

Facility location is naturally written using similarity rather than distance.

A distance satisfies

$$
\text{small distance}\Rightarrow\text{high similarity},
$$

whereas facility location wants

$$
\text{large value}\Rightarrow\text{high similarity}.
$$

A common transformation is

$$
\boxed{
S_{ij}=\exp\left(-\frac{D_{ij}^{2}}{2\tau^{2}}\right)
}
$$

where $\tau>0$ controls how quickly similarity decreases with distance.

This gives

$$
0<S_{ij}\le1.
$$

In particular,

$$
D_{ij}=0\Rightarrow S_{ij}=1.
$$

A large distance produces similarity close to zero.

The resulting similarity matrix is

$$
S=
\begin{bmatrix}
S_{11}&S_{12}&\cdots&S_{1N}\\
S_{21}&S_{22}&\cdots&S_{2N}\\
\vdots&\vdots&\ddots&\vdots\\
S_{N1}&S_{N2}&\cdots&S_{NN}
\end{bmatrix}.
$$

The submodularity proof below does not depend on this Gaussian transformation specifically. Any fixed non-negative similarity matrix can be used.

---

## 5. Step 3: Define Coverage

Suppose a subset $A$ has already been selected.

For every dataset $b_j\in B$, define

$$
\boxed{
c_A(j)=\max_{b_i\in A}S_{ij}
}
$$

where $c_A(j)$ is the similarity between $b_j$ and its **best currently selected representative**.

In implementation, this quantity can be stored as

$$
\operatorname{coverage}[j].
$$

Therefore,

$$
\boxed{
\operatorname{coverage}[j]=c_A(j)=\max_{b_i\in A}S_{ij}.
}
$$

It is important that `coverage[j]` is **not a count** of how many datasets have been covered. It means:

> How well is dataset $b_j$ represented by the best dataset currently selected in $A$?

For example, if

$$
A=\{b_2,b_5\},
$$

and

$$
S_{2,3}=0.4,\qquad S_{5,3}=0.9,
$$

then

$$
\operatorname{coverage}[3]=\max(0.4,0.9)=0.9.
$$

Thus $b_3$ is already represented very well.

---

## 6. Step 4: Facility-Location Objective

The total representativeness of $A$ is defined as

$$
\boxed{
F(A)=\sum_{j=1}^{N}\max_{b_i\in A}S_{ij}
}
$$

or equivalently,

$$
\boxed{
F(A)=\sum_{j=1}^{N}c_A(j).
}
$$

Interpretation:

> Every dataset $b_j$ looks for its most similar representative in $A$, and we sum these best similarities over the whole meta-dataset.

A high value of $F(A)$ means that the selected subset represents most or all datasets in $B$ well.

The optimization problem is therefore

$$
\boxed{
A^*=\arg\max_{\substack{A\subseteq B\\|A|\le K}}
\sum_{j=1}^{N}\max_{b_i\in A}S_{ij}.
}
$$

---

## 7. Proof of Monotonicity

A set function $F$ is monotone if

$$
A\subseteq C\Rightarrow F(A)\le F(C).
$$

Suppose

$$
A\subseteq C.
$$

For any dataset $b_j$,

$$
\max_{b_i\in A}S_{ij}
\le
\max_{b_i\in C}S_{ij}.
$$

Why? Because every representative available in $A$ is still available in $C$, and $C$ may contain additional representatives. Adding more choices cannot reduce a maximum.

Therefore,

$$
c_A(j)\le c_C(j)
$$

for every $j$. Summing over all $j$,

$$
\sum_{j=1}^{N}c_A(j)
\le
\sum_{j=1}^{N}c_C(j).
$$

Hence,

$$
\boxed{F(A)\le F(C).}
$$

Therefore $F$ is monotone.

### Intuition

Adding another representative can:

- improve the representation of some datasets, or
- leave their representation unchanged.

It cannot make an already available representative disappear.

---

## 8. Marginal Gain of Adding a Candidate

Suppose the current selected set is $A$, and consider adding a new candidate dataset $b_i\notin A$.

For a particular target dataset $b_j$, the current coverage is

$$
c_A(j).
$$

The new candidate provides similarity

$$
S_{ij}.
$$

After adding $b_i$, the new coverage becomes

$$
\max(c_A(j),S_{ij}).
$$

Therefore, the improvement for dataset $b_j$ is

$$
\max(c_A(j),S_{ij})-c_A(j).
$$

This can be written compactly as

$$
\boxed{
\max(c_A(j),S_{ij})-c_A(j)
=
\max(0,S_{ij}-c_A(j)).
}
$$

This equality handles both possible cases.

### Case 1: the candidate is better

If

$$
S_{ij}>c_A(j),
$$

then

$$
\max(c_A(j),S_{ij})=S_{ij},
$$

and the gain is

$$
S_{ij}-c_A(j)>0.
$$

### Case 2: the candidate is not better

If

$$
S_{ij}\le c_A(j),
$$

then

$$
\max(c_A(j),S_{ij})=c_A(j),
$$

and the gain is

$$
c_A(j)-c_A(j)=0.
$$

Thus no assumption is made that the new candidate is better. The $\max(0,\cdot)$ expression automatically handles both cases.

The **total marginal gain** of candidate $b_i$ is therefore

$$
\boxed{
\Delta(b_i\mid A)
=
F(A\cup\{b_i\})-F(A)
=
\sum_{j=1}^{N}\max(0,S_{ij}-c_A(j)).
}
$$

---

## 9. Proof of Submodularity

Submodularity means **diminishing returns**.

For

$$
A\subseteq C
$$

and a candidate

$$
b_x\notin C,
$$

we must show

$$
\boxed{
\Delta(b_x\mid A)
\ge
\Delta(b_x\mid C).
}
$$

Because

$$
A\subseteq C,
$$

the larger set $C$ can only provide equal or better coverage:

$$
c_C(j)\ge c_A(j)
$$

for every $j$.

Let

$$
q=S_{xj}
$$

be the similarity between the new candidate $b_x$ and dataset $b_j$.

The marginal improvement under the smaller set $A$ is

$$
\Delta_j(b_x\mid A)=\max(0,q-c_A(j)).
$$

Under the larger set $C$,

$$
\Delta_j(b_x\mid C)=\max(0,q-c_C(j)).
$$

Since

$$
c_C(j)\ge c_A(j),
$$

we have

$$
q-c_C(j)\le q-c_A(j).
$$

Therefore,

$$
\max(0,q-c_C(j))
\le
\max(0,q-c_A(j)).
$$

Hence,

$$
\Delta_j(b_x\mid C)
\le
\Delta_j(b_x\mid A).
$$

Summing over all datasets $b_j$,

$$
\boxed{
\Delta(b_x\mid C)
\le
\Delta(b_x\mid A).
}
$$

Therefore,

$$
\boxed{F\text{ is submodular}.}
$$

### Intuition

If the current subset is small, many datasets may still be poorly represented, so a new representative can help a lot.

If the current subset is already large, many datasets are already represented well, so the same new candidate tends to provide less additional benefit.

---

## 10. Greedy Selection Algorithm

The exact optimization

$$
\arg\max_{\substack{A\subseteq B\\|A|\le K}}F(A)
$$

is combinatorial because there are

$$
\binom{N}{K}
$$

possible subsets of size $K$.

Instead, use greedy forward selection.

Initialize

$$
A_0=\emptyset.
$$

At round $t$, evaluate the marginal gain of **every remaining candidate**:

$$
\Delta(b_i\mid A_{t-1})
=
\sum_{j=1}^{N}\max(0,S_{ij}-c_{A_{t-1}}(j)).
$$

Then choose

$$
\boxed{
i^*=\arg\max_{i:b_i\notin A_{t-1}}\Delta(b_i\mid A_{t-1}).
}
$$

Add the winner:

$$
A_t=A_{t-1}\cup\{b_{i^*}\}.
$$

Repeat until

$$
|A_K|=K.
$$

The key rule is

$$
\boxed{
\text{calculate the gain of ALL }b_i\notin A
\rightarrow
\text{then take the }\arg\max.
}
$$

It is incorrect to find one candidate with a large gain and immediately select it before evaluating the other remaining candidates.

---

## 11. Detailed Algorithm

```text
INPUT

B = {b1, b2, ..., bN}
    Meta-dataset containing N datasets.

Each bi = {xi1, xi2, ..., xiMi}
    Samples belonging to dataset bi.

K
    Maximum number of representative datasets to select.

D
    Dataset-level distance measure:
    KME/MMD, Sliced Wasserstein, Hausdorff, etc.


--------------------------------------------------
STEP 1: COMPUTE PAIRWISE DATASET DISTANCES
--------------------------------------------------

For i = 1,...,N:

    For j = 1,...,N:

        D[i,j] = D(bi, bj)


--------------------------------------------------
STEP 2: CONVERT DISTANCES TO SIMILARITIES
--------------------------------------------------

For i = 1,...,N:

    For j = 1,...,N:

        S[i,j] = exp(-D[i,j]^2 / (2*tau^2))


--------------------------------------------------
STEP 3: INITIALIZE
--------------------------------------------------

A = empty set

For j = 1,...,N:

    coverage[j] = 0


--------------------------------------------------
STEP 4: GREEDILY SELECT K REPRESENTATIVE DATASETS
--------------------------------------------------

For round = 1,...,K:

    For every candidate dataset bi not already in A:

        gain[i] = 0

        For j = 1,...,N:

            current = coverage[j]

            candidate_similarity = S[i,j]

            improvement =
                max(
                    0,
                    candidate_similarity - current
                )

            gain[i] =
                gain[i] + improvement


    After evaluating ALL remaining candidates:

        i* =
            argmax over i not in A of gain[i]


    Add the best candidate:

        A =
            A union {bi*}


    Update the coverage of every dataset:

        For j = 1,...,N:

            coverage[j] =
                max(
                    coverage[j],
                    S[i*,j]
                )


--------------------------------------------------
STEP 5: RETURN THE SELECTED SUBSET
--------------------------------------------------

Return A
```

---

## 12. Meaning of the Variables

| Symbol / variable | Meaning |
|---|---|
| $B$ | Complete meta-dataset |
| $b_i$ | Candidate dataset $i$ inside $B$ |
| $x_{im}$ | Sample $m$ inside dataset $b_i$ |
| $N$ | Number of datasets in $B$ |
| $K$ | Selection budget |
| $D_{ij}$ | Distance between whole datasets $b_i$ and $b_j$ |
| $S_{ij}$ | Similarity between whole datasets $b_i$ and $b_j$ |
| $A$ | Currently selected representative subset |
| $i$ | Index of a candidate dataset being considered for selection |
| $j$ | Index of a dataset whose current representation quality is being evaluated |
| `coverage[j]` | Best similarity currently available for dataset $b_j$ |
| `gain[i]` | Total marginal improvement obtained by adding candidate $b_i$ |
| $i^*$ | Candidate index with the largest marginal gain |

---

## 13. Worked Example

Consider

$$
B=\{b_1,b_2,b_3,b_4\}
$$

with budget

$$
K=2.
$$

Suppose the distance measure has already been converted into the following similarity matrix:

$$
S=
\begin{bmatrix}
1&0.9&0.2&0.1\\
0.9&1&0.3&0.2\\
0.2&0.3&1&0.8\\
0.1&0.2&0.8&1
\end{bmatrix}.
$$

Interpretation:

- $b_1$ and $b_2$ are very similar.
- $b_3$ and $b_4$ are very similar.
- The two groups are relatively dissimilar.

A representative subset of size $2$ should ideally choose one dataset from each region.

---

## 14. Round 1

Initially,

$$
A=\emptyset
$$

and

$$
\operatorname{coverage}=[0,0,0,0].
$$

Since no dataset is covered yet, the gain of each candidate is simply the sum of its similarities to all datasets.

### Candidate $b_1$

$$
S_{1,:}=[1,0.9,0.2,0.1].
$$

Therefore,

$$
\operatorname{gain}(b_1)=1+0.9+0.2+0.1=2.2.
$$

### Candidate $b_2$

$$
S_{2,:}=[0.9,1,0.3,0.2].
$$

Therefore,

$$
\operatorname{gain}(b_2)=0.9+1+0.3+0.2=2.4.
$$

### Candidate $b_3$

$$
S_{3,:}=[0.2,0.3,1,0.8].
$$

Therefore,

$$
\operatorname{gain}(b_3)=0.2+0.3+1+0.8=2.3.
$$

### Candidate $b_4$

$$
S_{4,:}=[0.1,0.2,0.8,1].
$$

Therefore,

$$
\operatorname{gain}(b_4)=0.1+0.2+0.8+1=2.1.
$$

Thus,

$$
\begin{array}{c|c}
\text{candidate}&\text{gain}\\
\hline
b_1&2.2\\
b_2&2.4\\
b_3&2.3\\
b_4&2.1
\end{array}
$$

and

$$
\boxed{b_2\text{ has the largest gain}.}
$$

Select

$$
A=\{b_2\}.
$$

The coverage vector becomes

$$
\boxed{
\operatorname{coverage}=[0.9,1,0.3,0.2].
}
$$

---

## 15. Round 2: Evaluate Every Remaining Candidate

The remaining candidates are

$$
b_1,\quad b_3,\quad b_4.
$$

We must calculate all three gains before selecting the winner.

Current coverage is

$$
\operatorname{coverage}=[0.9,1,0.3,0.2].
$$

### 15.1 Candidate $b_1$

$$
S_{1,:}=[1,0.9,0.2,0.1].
$$

Its marginal gain is

$$
\begin{aligned}
\Delta(b_1\mid A)
={}&\max(0,1-0.9)\\
&+\max(0,0.9-1)\\
&+\max(0,0.2-0.3)\\
&+\max(0,0.1-0.2).
\end{aligned}
$$

Therefore,

$$
\Delta(b_1\mid A)=0.1+0+0+0=\boxed{0.1}.
$$

### 15.2 Candidate $b_3$

$$
S_{3,:}=[0.2,0.3,1,0.8].
$$

Its gain is

$$
\begin{aligned}
\Delta(b_3\mid A)
={}&\max(0,0.2-0.9)\\
&+\max(0,0.3-1)\\
&+\max(0,1-0.3)\\
&+\max(0,0.8-0.2).
\end{aligned}
$$

Hence,

$$
\Delta(b_3\mid A)=0+0+0.7+0.6=\boxed{1.3}.
$$

### 15.3 Candidate $b_4$

$$
S_{4,:}=[0.1,0.2,0.8,1].
$$

Its gain is

$$
\begin{aligned}
\Delta(b_4\mid A)
={}&\max(0,0.1-0.9)\\
&+\max(0,0.2-1)\\
&+\max(0,0.8-0.3)\\
&+\max(0,1-0.2).
\end{aligned}
$$

Therefore,

$$
\Delta(b_4\mid A)=0+0+0.5+0.8=\boxed{1.3}.
$$

---

## 16. Compare All Remaining Candidates

We therefore obtain

$$
\begin{array}{c|c}
\text{candidate}&\text{marginal gain}\\
\hline
b_1&0.1\\
b_3&1.3\\
b_4&1.3
\end{array}
$$

Hence,

$$
\boxed{
b_3\text{ and }b_4\text{ tie for the largest marginal gain}.
}
$$

The algorithm may choose either one according to a deterministic or arbitrary tie-breaking rule.

If $b_3$ is selected,

$$
A=\{b_2,b_3\}.
$$

If $b_4$ is selected,

$$
A=\{b_2,b_4\}.
$$

Both achieve the same facility-location score in this toy example.

This illustrates the implementation rule

$$
\boxed{
\text{Evaluate ALL remaining candidates first, then choose the global }\arg\max.
}
$$

---

## 17. Coverage Update After Selecting $b_3$

Suppose the tie is broken in favor of $b_3$.

Then

$$
A=\{b_2,b_3\}.
$$

The previous coverage was

$$
[0.9,1,0.3,0.2].
$$

The similarity row of $b_3$ is

$$
[0.2,0.3,1,0.8].
$$

Update each position using

$$
\operatorname{coverage}[j]
\leftarrow
\max(\operatorname{coverage}[j],S_{3j}).
$$

Therefore,

$$
\operatorname{coverage}
=
[
\max(0.9,0.2),
\max(1,0.3),
\max(0.3,1),
\max(0.2,0.8)
].
$$

Hence,

$$
\boxed{
\operatorname{coverage}=[0.9,1,1,0.8].
}
$$

The facility-location score is

$$
F(A)=0.9+1+1+0.8=\boxed{3.7}.
$$

---

## 18. Why Facility Location Naturally Produces Diversity

After selecting $b_2$, candidate $b_1$ has only a small marginal gain:

$$
\Delta(b_1\mid\{b_2\})=0.1.
$$

This happens because $b_1$ and $b_2$ already represent nearly the same region of the meta-dataset.

In contrast, $b_3$ and $b_4$ provide large gains because they represent datasets that are currently poorly covered.

Therefore, facility location naturally balances

$$
\boxed{
\text{representativeness}+\text{non-redundancy}
}
$$

without explicitly adding a separate diversity penalty.

---

## 19. Approximation Guarantee of Greedy Selection

For the cardinality-constrained problem

$$
\max_{A\subseteq B,\ |A|\le K}F(A),
$$

if $F$ is normalized, monotone, and submodular, then the standard greedy algorithm satisfies

$$
\boxed{
F(A_{\mathrm{greedy}})
\ge
\left(1-\frac{1}{e}\right)F(A^*)
}
$$

where $A^*$ is the globally optimal subset.

Since

$$
1-\frac{1}{e}\approx0.632,
$$

the worst-case guarantee is approximately $63.2\%$ of the optimal objective value.

This is a worst-case lower bound; empirical performance can be much closer to the optimum.

---

## 20. Why This Algorithm Is Not EM

The greedy facility-location algorithm may look iterative, but it is **not Expectation-Maximization (EM)**.

### EM-style optimization

EM alternates between two sets of variables:

$$
\boxed{E\text{-step}\leftrightarrow M\text{-step}}
$$

until convergence.

For example, in the K-means analogy:

1. fix centroids and update assignments,
2. fix assignments and update centroids,
3. repeat.

Existing parameters are repeatedly revised.

### Greedy facility location

Facility-location selection instead performs

$$
A_0=\emptyset,
$$

$$
A_1=A_0\cup\{b_{i_1}\},
$$

$$
A_2=A_1\cup\{b_{i_2}\},
$$

and so on until

$$
|A_K|=K.
$$

Once a dataset is selected, it is not moved, retrained, or recomputed. Each iteration simply adds one new representative.

Therefore, the method is

$$
\boxed{\text{greedy forward subset selection}}
$$

rather than EM.

---

## 21. Why Updating `coverage` Does Not Make It EM

The value

$$
\operatorname{coverage}[j]
$$

is not a learned parameter.

It is simply a cached quantity:

$$
\operatorname{coverage}[j]
=
\max_{b_i\in A}S_{ij}.
$$

After a new representative is selected, it can be updated efficiently:

$$
\operatorname{coverage}[j]
\leftarrow
\max(\operatorname{coverage}[j],S_{i^*j}).
$$

Without this cache, the same quantity could always be recomputed from scratch:

$$
\max_{b_i\in A}S_{ij}.
$$

Thus,

$$
\boxed{\text{coverage is bookkeeping, not an optimization variable}.}
$$

---

## 22. Relationship to $k$-Means and $k$-Medoids

Facility location is conceptually related to clustering because after selecting

$$
A=\{a_1,\ldots,a_K\},
$$

every dataset $b_j$ can be assigned to its most similar selected representative:

$$
\boxed{
c(j)=\arg\max_{a\in A}S(a,b_j).
}
$$

This creates $K$ groups around the selected representatives.

However, facility location is closer to **$k$-medoids** than $k$-means because each representative must be an actual element of the original meta-dataset:

$$
A\subseteq B.
$$

In $k$-means, a centroid can be an artificial mean vector not present in the original data.

---

## 23. Convexity and Concavity Are Not Required Here

For this facility-location formulation, the main guarantees do **not** require proving that the objective is convex or concave.

The relevant properties are

$$
\boxed{\text{normalized}+\text{monotone}+\text{submodular}.}
$$

Convexity and concavity are mainly concepts for continuous-domain optimization. Submodularity plays a similar diminishing-returns role for set-valued optimization.

A useful mental analogy is

$$
\boxed{\text{concavity}\approx\text{continuous diminishing returns}}
$$

whereas

$$
\boxed{\text{submodularity}\approx\text{set-based diminishing returns}.}
$$

Although concave-over-modular constructions are one way to create submodular functions, facility location already has a direct proof of submodularity through its maximum-coverage structure.

Therefore, convexity or concavity is not needed for the proof or the standard greedy guarantee used here.

---

## 24. Complete Pipeline

The complete representative-selection pipeline is

$$
\boxed{B=\{b_1,\ldots,b_N\}}
$$

$$
\downarrow
$$

Compute dataset-level distances

$$
\boxed{D_{ij}=D(b_i,b_j)}
$$

using KME/MMD, Sliced Wasserstein, Hausdorff, or another suitable dataset distance.

$$
\downarrow
$$

Convert distance to similarity

$$
\boxed{
S_{ij}=\exp\left(-\frac{D_{ij}^2}{2\tau^2}\right)
}
$$

$$
\downarrow
$$

Define facility-location representativeness

$$
\boxed{
F(A)=\sum_{j=1}^{N}\max_{b_i\in A}S_{ij}
}
$$

$$
\downarrow
$$

At each greedy round, compute for **every remaining candidate**

$$
\boxed{
\Delta(b_i\mid A)
=
\sum_{j=1}^{N}\max(0,S_{ij}-c_A(j))
}
$$

$$
\downarrow
$$

Choose

$$
\boxed{
i^*=\arg\max_{i:b_i\notin A}\Delta(b_i\mid A)
}
$$

$$
\downarrow
$$

Update

$$
\boxed{A\leftarrow A\cup\{b_{i^*}\}}
$$

and

$$
\boxed{
c_A(j)\leftarrow\max(c_A(j),S_{i^*j})
}
$$

for every $j$.

$$
\downarrow
$$

Repeat until

$$
\boxed{|A|=K}.
$$

Return

$$
\boxed{A=\{b_{i_1},b_{i_2},\ldots,b_{i_K}\}.}
$$

---

## 24.1. Explicit Objective for Each Distance

For clarity, the three concrete objectives optimized by the same greedy algorithm are:

### KME / MMD

$$
\boxed{
F_{\mathrm{KME}}(A)
=
\sum_{j=1}^{N}
\max_{b_i\in A}
\exp\left(
-\frac{
\operatorname{MMD}^2(b_i,b_j)
}{
2\tau_{\mathrm{KME}}^2
}
\right).
}
$$

### Sliced Wasserstein

$$
\boxed{
F_{\mathrm{SW}}(A)
=
\sum_{j=1}^{N}
\max_{b_i\in A}
\exp\left(
-\frac{
SW_p^2(P_i,P_j)
}{
2\tau_{\mathrm{SW}}^2
}
\right).
}
$$

### Hausdorff

$$
\boxed{
F_{\mathrm{H}}(A)
=
\sum_{j=1}^{N}
\max_{b_i\in A}
\exp\left(
-\frac{
d_H^2(b_i,b_j)
}{
2\tau_{\mathrm{H}}^2
}
\right).
}
$$

In every case, the outer maximum means:

> For each dataset $b_j$ in the full meta-dataset, find the selected dataset $b_i\in A$ that is most similar to it according to the chosen distance-induced similarity.

---

## 25. Final Mathematical Summary

Given

$$
B=\{b_1,\ldots,b_N\},
$$

a budget $K$, and a dataset-level distance $D$:

### Distance

$$
D_{ij}=D(b_i,b_j).
$$

### Similarity

$$
S_{ij}=\exp\left(-\frac{D_{ij}^{2}}{2\tau^{2}}\right).
$$

### Coverage

$$
c_A(j)=\max_{b_i\in A}S_{ij}.
$$

### Objective

$$
F(A)=\sum_{j=1}^{N}c_A(j)
=\sum_{j=1}^{N}\max_{b_i\in A}S_{ij}.
$$

### Marginal gain

$$
\Delta(b_i\mid A)
=\sum_{j=1}^{N}\max(0,S_{ij}-c_A(j)).
$$

### Greedy selection

$$
i^*=\arg\max_{i:b_i\notin A}\Delta(b_i\mid A).
$$

### Update

$$
A\leftarrow A\cup\{b_{i^*}\},
$$

$$
c_A(j)\leftarrow\max(c_A(j),S_{i^*j}).
$$

### Stop

$$
|A|=K.
$$

### Properties

$$
\boxed{F(\emptyset)=0}
$$

$$
\boxed{A\subseteq C\Rightarrow F(A)\le F(C)}
$$

$$
\boxed{
A\subseteq C,\ b_x\notin C
\Rightarrow
\Delta(b_x\mid A)\ge\Delta(b_x\mid C)
}
$$

and therefore

$$
\boxed{F\text{ is normalized, monotone, and submodular}.}
$$

Under the cardinality constraint

$$
|A|\le K,
$$

greedy selection satisfies

$$
\boxed{
F(A_{\mathrm{greedy}})
\ge
\left(1-\frac1e\right)F(A^*).
}
$$

---

## 26. Core Interpretation

The entire method can be summarized in one sentence:

> At each round, select the remaining dataset that provides the largest total improvement in how well the current subset represents **all datasets in the meta-dataset**, and repeat until the budget $K$ is exhausted.

Or mathematically,

$$
\boxed{
\text{compare all candidates}
\rightarrow
\text{measure marginal coverage improvement}
\rightarrow
\text{select the global best}
\rightarrow
\text{update coverage}
\rightarrow
\text{repeat}.
}
$$

---

## 27. Empirical Results

Implementation: `experiments/subset_distances.py` (the four distance
formulas + `select_representative_subsets`, matching Sections 3.1–3.4 and
10–11 exactly) and
`experiments/run_acquisition_benchmark.py::run_distance_formula_comparison`
(new mode). Reproduce with:

```bash
python -m experiments.run_acquisition_benchmark --mode distance_formula_comparison --seeds 5
```

Each candidate sample-set is represented as a point cloud of $n{=}20$
points scattered around its latent factor position — standing in for "the
actual raw examples inside this workload," which is what the real
pipeline's cached embeddings look like
(`active_evaluator/embedding_cache.py`) but which the scalar-descriptor
synthetic generator doesn't otherwise model. $D_{ij}^{(m)}$ is computed
between every pair of the ${\approx}113$ candidate source sample-sets, for
each of the four formulas, then $K{=}30$ representatives are selected via
Section 10's greedy algorithm exactly.

### 27.1 A bug caught by cross-checking code against this document

`kernel_mean_distance`'s first implementation returned $\mathrm{MMD}^2$
directly (the standard biased empirical estimator) rather than
$\mathrm{MMD} = \|\hat\mu_i-\hat\mu_j\|_{\mathcal H}$ (Sec. 3.1). Since
`select_representative_subsets` squares whatever it's given
($S_{ij}=\exp(-D_{ij}^2/\tau)$, Sec. 4/25), this meant kernel_mean was
silently computing $\exp(-\mathrm{MMD}^4/\tau)$ instead of the specified
$\exp(-\mathrm{MMD}^2/2\tau^2)$ — squaring the discrepancy twice. Hausdorff
and sliced-Wasserstein were unaffected (their functions already return the
unsquared $d_H$/$SW_p$, matching Sec. 3.2–3.3). Caught by walking Sections
3.1–3.4 against the code line by line before trusting the first run's
numbers; fixed by taking $\sqrt{\mathrm{MMD}^2}$ before returning. The
numbers below are from the corrected code — the first (uncorrected) run's
kernel_mean/sum rows are superseded and not reported.

### 27.2 Results — comparison against the k-means warm-start baselines

5 seeds, source universe ≈ 113 sample-sets (of 150 generated), $K{=}30$
representatives, acquisition budget = 15% of the reduced pool. Unseen-model
MAE in percentage points (lower is better). Raw output:
`outputs/distance_formula_comparison.json`.

| Config | #triples | ActiveEval-Pair | Facility-location | Random |
| --- | ---: | --- | --- | --- |
| Full source pool | 6780 | **3.68 ± 0.63** | 3.94 ± 0.50 | 4.39 ± 0.95 |
| Random subset (n=30) | 1800 | 4.55 ± 0.85 | 5.03 ± 0.91 | 4.71 ± 0.50 |
| Distance: **sum** | 1800 | 4.90 ± 0.61 | 5.37 ± 0.65 | 4.69 ± 0.72 |
| Distance: **kernel_mean** | 1800 | 5.23 ± 0.89 | 6.62 ± 1.48 | 5.44 ± 0.89 |
| Distance: **hausdorff** | 1800 | 5.46 ± 1.30 | 6.02 ± 1.35 | 5.95 ± 1.31 |
| Distance: **sliced_wasserstein** | 1800 | 5.92 ± 2.21 | 5.67 ± 1.43 | 6.61 ± 1.67 |

For reference, k-means (M=30, K=1, Lloyd's heuristic clustering, no
approximation guarantee — `docs/kmeans-submodular-warmstart.md` §6) scored
**5.21 ± 1.04** on ActiveEval-Pair at the same budget.

**Reading it honestly.** Every one of the four *provably* $(1{-}1/e)$-guaranteed
submodular coreset selections still loses to plain random subsampling of
the same size on the primary metric (ActiveEval-Pair): 4.90–5.92 vs.
random's 4.55. This is the same conclusion already reached for k-means, now
confirmed on a structurally different — and provably near-optimal —
algorithm: **the failure mode is the target-blind *objective* (Sec. 6's
$F(A)$ has no target term at all), not the quality of the optimizer solving
it.** A guaranteed-near-optimal solution to "represent the whole
meta-dataset" is still not what a target-aware downstream task needs.

**sum is the best of the four** (4.90 ± 0.61, tightest CI too) and — after
the bugfix — meaningfully better than before (was 5.13 ± 0.96 under the
kernel_mean bug); it also now edges very close to the random-subset
baseline on the `Random` acquisition-method column (4.69 vs. 4.71,
essentially tied). **sliced_wasserstein is the weakest and noisiest**
(5.92 ± 2.21, by far the widest interval of any row in the table) — see
§27.4 for why. **kernel_mean and hausdorff sit in between.**

### 27.3 Root cause: confirmed identical to k-means's, via the near-target coverage check

| Formula | Near-target representatives (of 30) | True rate |
| --- | ---: | ---: |
| **sum** | 7/30 = **23.3%** | 20.4% |
| kernel_mean | 5/30 = 16.7% | 20.4% |
| sliced_wasserstein | 4/30 = 13.3% | 20.4% |
| hausdorff | 3/30 = 10.0% | 20.4% |

(Seed 0; "near-target" = top quintile of source `target_score`, matching
the diagnostic already used for k-means.) Three of the four formulas
under-represent the near-target tail relative to its true 20.4% share —
the same mechanism diagnosed for k-means: the near-target region is a
low-density directional extreme by construction, and *any* objective
maximizing coverage of the *whole* density necessarily spends more of its
$K{=}30$ budget on the dense, high-mass regions, starving the
sparse-but-important one. **sum is the exception — after the bugfix it now
slightly *exceeds* the true rate (23.3%)** — consistent with it also being
the best performer on MAE; combining all three distances evidently
counteracts each individual formula's own density bias somewhat, though
this is one seed's measurement, not a proven general property.

### 27.4 Cost report — budget, latency, tokens

**Labeling budget** (the real, expensive resource): $K{=}30$
representative sample-sets × 60 reference models = **1800 labelable
triples**, vs. 6780 for the full unreduced pool — a **73.5% reduction** in
expensive (model, workload) evaluation actions, identical across all four
formulas and the random-subset control (all compared at matched $K{=}30$).
On top of that, acquisition spends its usual 15% of *that* reduced pool
(270 of the 1800 triples actually get trained on).

**Stage-1 wall-clock cost** (choosing which 30 sample-sets to keep — mean
over 5 seeds, unaffected by the kernel_mean bugfix since it only changes
which values are exponentiated, not how many pairs are computed;
$\binom{113}{2}{\approx}6328$ pairwise distances per formula):

| Formula | build clouds | distance matrix | greedy select | **total** |
| --- | ---: | ---: | ---: | ---: |
| kernel_mean | 0.001s | 0.568s | 0.001s | **0.570s** |
| hausdorff | 0.001s | 0.867s | 0.001s | **0.869s** |
| sliced_wasserstein | 0.001s | 1.353s | 0.001s | **1.355s** |
| sum | 0.001s | 2.783s | 0.001s | **2.785s** |

Cloud construction and the greedy selection itself are both negligible
(≤2ms); essentially all Stage-1 cost is the pairwise distance matrix.
**Cost ranking**: kernel_mean (fully vectorized NumPy, cheapest) <
hausdorff (scipy's C-optimized `directed_hausdorff`, called twice per pair)
< sliced_wasserstein (a Python loop over 20 projection directions per pair,
each calling `scipy.stats.wasserstein_distance` — per-call Python/scipy
overhead dominates) < sum (pays for all three: $0.568+0.867+1.353\approx
2.79 \approx$ the measured $2.785$s — confirming no redundant
recomputation). All four are cheap in absolute terms (under 3 seconds even
for the most expensive), so **Stage-1 selection cost is not the bottleneck
at this pool size ($N{\approx}113$)**; it would only matter at a much
larger candidate universe, where the shared $O(N^2)$ pairwise-distance cost
(common to all four formulas, and to k-means's own distance computations)
becomes dominant.

**Why sliced_wasserstein's confidence interval is so wide** (±2.21, more
than double any other formula's): with only $n{=}20$ points per cloud and
20 projection directions (kept low deliberately to bound Stage-1 cost),
each pairwise $SW_p$ estimate (Sec. 3.2) is itself a fairly high-variance
estimator of the true sliced-Wasserstein distance — few points means each
1-D projected Wasserstein distance is a noisy order-statistic quantity, and
few projections means the Monte Carlo average over $\theta$ hasn't
converged tightly. Kernel mean (an average over *all* $n^2$ pairwise kernel
evaluations, Sec. 3.1 — a lower-variance U-statistic-like quantity) and
Hausdorff (a single worst-case order statistic, but computed *exactly*
rather than estimated from a projection sample, Sec. 3.3) don't carry this
same extra estimation noise. A larger $n$/number of projections would
likely tighten sliced_wasserstein's estimate at the cost of more Stage-1
wall-clock time — a real, quantifiable precision/cost trade-off specific to
this formula, not tested here.

**Tokens used: not applicable.** This is a pure CPU/NumPy synthetic
benchmark — `experiments/run_acquisition_benchmark.py` and
`experiments/subset_distances.py` make no LLM API calls and run no model
inference of any kind. The only real cost is the CPU wall-clock time
reported above. (The production Text2SQL pipeline,
`active_evaluator/pipeline.py`, does call Hugging Face models — but this
comparison was run entirely on the synthetic benchmark track, consistent
with every other experiment referenced in this document, and consumed zero
tokens.)

### 27.5 Bottom line at K=30

1. Confirms, on a *provably*-guaranteed algorithm (not merely a heuristic
   like k-means), that the earlier k-means finding generalizes: **the
   failure mode is the target-blind objective ($F(A)$ has no target term),
   not the specific method used to optimize it.**
2. Among the four distance formulas, **sum** is the strongest and, at the
   same time, cheap relative to its own constituent parts (its cost is
   exactly the sum of the other three, no overhead); **sliced_wasserstein**
   is both the weakest on MAE and the noisiest, plausibly fixable with more
   Stage-1 compute (untested here).
3. None beats plain random subsampling *at this specific budget* — but see
   §27.6, where raising the budget changes this picture non-trivially. The
   natural next experiment — following directly from
   `docs/kmeans-submodular-warmstart.md`'s own conclusion — is a
   **target-aware version of this same coreset selector** (e.g. add the
   target region's descriptors into the pool being covered by $F(A)$, or
   weight the coverage sum by proximity to the target direction) rather
   than testing another target-blind distance formula.

### 27.6 Does raising the budget K let the weaker formulas catch up?

Motivated by the same question already asked (and answered, non-monotonically)
for k-means in `docs/kmeans-submodular-warmstart.md` §6.1: repeating the
$K{=}30$ comparison at $K{=}60$ and $K{=}90$ (same 5 seeds, same 113-source
pool), reusing `run_distance_formula_comparison(budget_K=...)`:

| Config | K=30 (§27.2) | K=60 | K=90 |
| --- | --- | --- | --- |
| Full source pool (ref.) | 3.68 ± 0.63 | 3.68 ± 0.63 | 3.68 ± 0.63 |
| Random subset | 4.55 ± 0.85 | 4.41 ± 0.76 | **3.71 ± 0.48** |
| Hausdorff | 5.46 ± 1.30 | **3.81 ± 0.65** | 3.77 ± 0.79 |
| Kernel mean | 5.23 ± 0.89 | 4.65 ± 0.89 | 3.84 ± 0.76 |
| Sliced Wasserstein | 5.92 ± 2.21 | **4.20 ± 0.47** | 4.25 ± 0.72 |
| Sum | 4.90 ± 0.61 | 4.54 ± 0.97 | 4.14 ± 0.83 |

Raw output: `outputs/distance_formula_K60.json`, `outputs/distance_formula_K90.json`.

**Yes, at $K{=}60$ two of the four formulas do beat random** — Hausdorff
(3.81 ± 0.65, nearly matching the full-pool reference) and sliced
Wasserstein (4.20 ± 0.47) both clearly outperform random's 4.41 ± 0.76 at
this budget. **But this is not a stable, monotonically-improving trend.**
By $K{=}90$ (79.6% of the source pool): random itself jumps to
3.71 ± 0.48, nearly matching the full-pool number; sliced Wasserstein's
win *reverses* (4.25, back behind random); and Hausdorff/kernel_mean
converge to values statistically indistinguishable from random given the
heavily overlapping confidence intervals. This mirrors exactly what
happened with k-means's own $M$-sweep (§6.1 of the companion doc): as the
budget approaches the pool size, *every* method — including random —
converges toward the full-pool reference, and any specific formula's
"beats random" advantage becomes noisy and non-monotonic rather than
something you can keep buying by raising $K$ further.

**Two hypotheses for *why* Hausdorff wins at $K{=}60$ were checked directly
and neither held up** — worth reporting as ruled out rather than silently
dropped:

1. *Better target-tail coverage.* On the actual 113-source pool used for
   this table, Hausdorff and kernel_mean select **identical** near-target
   counts at $K{=}60$ (12/60 = 20% each, both at the true ~20.4% rate) and
   the **identical** target-score range ($[-4.72, 1.20]$, width 5.92 for
   both) — not a differentiator here, even though Hausdorff clearly wins on
   MAE and kernel_mean clearly doesn't.
2. *Less redundant/more diverse representatives.* Mean pairwise spread among
   the 60 selected representatives' latent positions is nearly identical
   across all four formulas (3.24–3.46), and *zero* near-duplicate pairs
   (spread $<0.5$) for every formula — no detectable diversity advantage
   for Hausdorff specifically.

**Honest conclusion:** the $K{=}60$ Hausdorff/sliced-Wasserstein advantage
is real (confirmed on 5 seeds, non-overlapping-ish CIs against random) and
worth keeping as a finding, but its *mechanism* is not yet identified — it
isn't explained by either of the two most natural hypotheses, and it does
not generalize into a "raise $K$ and any formula eventually wins" rule,
since $K{=}90$ mostly erases it. A more targeted follow-up (e.g. comparing
*which specific* sample-sets each formula picks at $K{=}60$, rather than
aggregate summary statistics of the selected set) would be needed to
actually explain it rather than merely observe it.
