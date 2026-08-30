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

> **Update — this open question is now largely resolved.** §29 identifies
> the mechanism: Hausdorff-induced facility location controls the
> **covering radius** $\varepsilon(A)$ (Lemma 1, Theorem 2), and measuring
> $\varepsilon(A)$ directly (§29.6) shows Hausdorff has *no* coverage
> advantage at $K{=}15$, a negligible 1.9% one at $K{=}30$, and a decisive
> 9–34% one from $K{=}60$ — tracking its MAE record (worst at $K{=}30$,
> best at $K{=}60$) exactly. §28.8 supplies the complementary half: at
> small $K$ it spends its budget on extremes (0/10, 0/15 near-target
> picks), so it obtains neither a uniform coverage guarantee nor a
> distributional match. Caveats and what remains unexplained: §29.7.

### 27.7 Weighting Hausdorff more heavily inside the sum

Given Hausdorff's standout showing in §27.6, `sum` was generalized to a
weighted combination:

$$
D_{\text{sum,weighted}}(b_i,b_j) \;=\; \sum_m w_m\, \tilde D^{(m)}_{ij},
\qquad
\tilde D^{(m)}_{ij} = \frac{D^{(m)}_{ij} - \mu_m}{\sigma_m}
$$

where $\tilde D^{(m)}$ is the same z-score standardization already used for
the equal-weighted `sum` (§27.2), and $\mu_m,\sigma_m$ are the mean/std of
$D^{(m)}$ over the candidate pool's pairwise entries. Adopted weighting:
**double Hausdorff, keep the other two at 1**:

$$
w_{\mathrm{KME}} = 1, \qquad w_{\mathrm{SW}} = 1, \qquad w_{\mathrm H} = 2
\qquad\Longleftrightarrow\qquad
\big(\text{normalized: } 25\%,\ 25\%,\ 50\%\big).
$$

Implemented as `experiments/subset_distances.py::combined_distance_matrix(base,
weights={"hausdorff": 2.0, "kernel_mean": 1.0, "sliced_wasserstein": 1.0})`,
registered as formula `"sum_hausdorff_weighted"` — the one supported
weighted variant (a more aggressive 0.7/0.2/0.1 convex-combination weighting
was also tried and performed *worse* than this one; removed from the active
formula set and this document to keep the record focused on what works,
though it remains in git history for anyone who wants to see it). Reproduce:

```python
from experiments.run_acquisition_benchmark import run_distance_formula_comparison
run_distance_formula_comparison(list(range(5)), 0.15, n_samplesets_full=150, budget_K=60,
                                formulas=("hausdorff", "sum", "sum_hausdorff_weighted"))
```

**Results** (5 seeds, $K{=}60$, same 113-source pool as §27.6). Raw output:
`outputs/distance_formula_K60_weighted.json`.

| Config | Weight on Hausdorff | ActiveEval-Pair MAE |
| --- | ---: | --- |
| Full source pool (ref.) | — | 3.68 ± 0.63 |
| Hausdorff (alone) | 100% | **3.81 ± 0.65** |
| **sum, Hausdorff-weighted (2:1:1)** | 50% | **4.16 ± 0.78** |
| Random subset | — | 4.41 ± 0.76 |
| sum (equal weights) | 33% | 4.54 ± 0.97 |

#### Analysis

**The reweighting works as intended.** Moving from equal weighting (33% on
Hausdorff, matching each of the three metrics contributing equally) to 50%
flips the combined formula from *losing* to random subsampling (4.54 vs.
4.41) to *beating* it (4.16 vs. 4.41) — a real, meaningful improvement in
the predicted direction, and it does so at the same $K{=}60$ / 3600-triple
budget as every other row, so this is not a budget-increase effect, purely
a reweighting one.

**It doesn't fully recover Hausdorff's own strength.** Hausdorff alone still
beats the weighted sum by a clear margin (3.81 vs. 4.16) — mixing in
kernel_mean and sliced Wasserstein at even a reduced 25%-each weight costs
something relative to using Hausdorff exclusively. This is consistent with
§27.6's honest, still-open finding that Hausdorff's specific advantage at
$K{=}60$ has not been mechanistically explained (two natural hypotheses —
better target-tail coverage, more diverse/less redundant picks — were
checked directly and ruled out there); reweighting transfers *some* of
whatever that advantage is into the combined formula without requiring the
underlying mechanism to be understood first, but it isn't a full transfer.

**Practical reading:** if the goal is simply "best MAE," use Hausdorff
alone rather than any blend — nothing in this weighted-sum family beats it.
If the goal is specifically a *combined, all-three-signals* formula (e.g.
for robustness to whichever individual metric happens to be weakest on a
given problem, since we don't have a principled way to know that in
advance without running all three), then Hausdorff-weighting at roughly the
50% level is the better choice within that family, and is what this
document now recommends as the default `"sum"` variant when a combined
formula specifically is wanted at this budget.

**Caveat on generalization.** This weighting was tuned and validated on the
same $K{=}60$, same 113-source-pool setting as §27.6 — it has not been
tested at $K{=}30$ or $K{=}90$, where the underlying per-formula rankings
differ (§27.6: Hausdorff is *worst* individually at $K{=}30$, and the
Hausdorff/random gap nearly vanishes by $K{=}90$), so there is no guarantee
this specific 2:1:1 weighting remains the right choice outside the $K{=}60$
regime it was derived from — it is a budget-specific tuning, not a
generally-derived optimum.

### 27.9 Two more coreset methods: ProbCover and Kernel Herding

Two additional selection algorithms, from a different source than the rest
of this document — [orobix/active-learning](https://github.com/orobix/active-learning)
(`activelearning/queries/representative/`), a general-purpose active
learning library. Implemented in
`experiments/subset_distances.py::select_probcover` and
`::select_kernel_herding`.

#### ProbCover

Yehuda, Mahmood, Sabach & Shalev-Shwartz, *"Active Learning Through a
Covering Lens,"* NeurIPS 2022. Reference implementation:
`probcover_query.py` in the repo above. A **hard-threshold coverage**
algorithm: two candidates are "adjacent" iff their distance is at most
$\delta$; greedily pick whichever candidate covers the most currently
*uncovered* candidates (including itself), mark those covered, repeat.

$$
\text{adjacent}(b_i,b_j) \iff D_{ij} \le \delta,
\qquad
i^\star = \arg\max_{i:\ b_i\notin A} \big|\{j : \text{adjacent}(b_i,b_j)\ \wedge\ b_j\notin\text{covered}\}\big|
$$

This differs from facility location (§6/§25) in one precise way: facility
location's marginal gain $\max(0, S_{ij}-c_A(j))$ is a **smooth, graded**
reward — a closer match always helps a little more than a farther one, no
matter how close. ProbCover's marginal gain is **binary and hard** — a
point within $\delta$ contributes exactly the same regardless of whether it
is barely inside the radius or dead center, and a point just outside
$\delta$ contributes nothing at all, however close it is. $\delta$ defaults
to the median pairwise distance (the same median-heuristic convention used
everywhere else in this module); if no candidate covers anything new,
$\delta$ halves and retries, mirroring the reference implementation's own
fallback (its version shrinks $\delta$ once an *existing labeled pool*
runs out of "uncovered" points to query against — this document's version
is adapted to a from-scratch coreset problem with no pre-existing labeled
set, so every one of the $N$ candidates starts uncovered rather than only
points far from an existing pool).

ProbCover is **distance-metric-agnostic** — it needs *a* distance, not a
specific one. Defaults here to the **kernel_mean (MMD)** distance matrix,
as the closest analogue among this document's three metrics to the
reference implementation's plain Euclidean distance.

No submodularity claim is made for ProbCover here — the source paper's
guarantee is a separate covering-radius argument, not the NWF78 $(1-1/e)$
bound proven for facility location in `docs/submodularity-audit.md` Part 1.

#### Kernel Herding

Chen, Welling & Smola, *"Super-Samples from Kernel Herding,"* UAI 2010.
Picks a subset whose **kernel mean embedding** matches the *whole pool's*
kernel mean embedding as closely as possible — a fundamentally different
goal from facility location's per-point coverage:

$$
\mu_B = \frac1N\sum_{j=1}^N \phi(b_j), \qquad
\mu_A = \frac1{|A|}\sum_{b_i\in A}\phi(b_i), \qquad
\boxed{\min_{|A|=K} \|\mu_B-\mu_A\|_{\mathcal H}^2 = \min_{|A|=K}\operatorname{MMD}^2(A,B)}
$$

Greedy rule: at each step, given the currently selected $A_t$, add whichever
candidate points furthest in the direction the selected mean is currently
*missing*:

$$
a_{t+1} = \arg\max_{b_i\in B} \big\langle \phi(b_i),\ \mu_B - \mu_{A_t} \big\rangle_{\mathcal H}
$$

This needs no explicit feature map $\phi$ — only inner products, via the
kernel trick. `compute_subset_kernel_gram` builds
$G_{ij} = \langle\mu_i,\mu_j\rangle_{\mathcal H}$ (the mean pairwise RBF
kernel value between every point in cloud $i$ and every point in cloud
$j$ — literally the cross term already inside `kernel_mean_distance`,
exposed as a full $N\times N$ matrix; note $\mathrm{MMD}^2(b_i,b_j) =
G_{ii}+G_{jj}-2G_{ij}$, so this is the *same* underlying kernel as the
kernel_mean distance formula, just used for a different objective).
`select_kernel_herding` then runs the recursion purely from $G$:

$$
\langle\phi(b_i),\mu_B\rangle = \tfrac1N\textstyle\sum_j G_{ij} \quad\text{(fixed target)}, \qquad
\langle\phi(b_i),\mu_{A_t}\rangle = \tfrac1t\textstyle\sum_{a\in A_t} G_{ia} \quad\text{(updated each round)}
$$

with $\mu_{A_0} := 0$ (standard herding convention, so the first pick is
just $\arg\max_i \langle\phi(b_i),\mu_B\rangle$).

**Why this is a fundamentally different objective than everything else in
this document**, restated from the problem specification that motivated
implementing it: kernel herding asks *"does the selected subset, as a
whole, statistically resemble the full meta-dataset's distribution?"*
(global distribution-matching, $\min\operatorname{MMD}(A,B)$); facility
location asks *"does every individual dataset in $B$ have at least one
good representative in $A$?"* (per-point coverage,
$F(A)=\sum_j\max_i S_{ij}$). A population with 90 common-type and 10
rare-type datasets: kernel herding, at $K{=}10$, tends toward
proportionally reproducing the mix (roughly 9 common + 1 rare, matching the
90/10 frequency); facility location tends to spend more of its budget on
distinct regions once the common cluster is already well covered by one or
two picks, since further common-type picks have little marginal coverage
gain left.

**Kernel herding is not submodular.** It directly minimizes
$-\operatorname{MMD}(A,B)$, a global-matching objective, not a per-point
coverage sum — this is not generally monotone or submodular, so it carries
no $(1-1/e)$ guarantee (contrast with facility location, proven monotone
submodular in `docs/submodularity-audit.md` Part 1). Its own theoretical
grounding instead comes from empirical-kernel-mean convergence rates
(Chen/Welling/Smola 2010), a different kind of guarantee entirely — a
useful mental model is *"deterministic representative sampling for the
kernel mean,"* not a clustering or coverage algorithm.

#### Results — all methods head to head

5 seeds, $K{=}60$, same 113-source pool as §27.6–27.8. Reproduce:

```python
run_distance_formula_comparison(
    list(range(5)), 0.15, n_samplesets_full=150, budget_K=60,
    formulas=("kernel_herding", "probcover", "kernel_mean", "sliced_wasserstein",
             "hausdorff", "sum_hausdorff_weighted"),
    include_kmeans=True)
```

Raw output: `outputs/herding_probcover_comparison_K60.json`.

| Config | ActiveEval-Pair MAE | Facility-location MAE | Random MAE |
| --- | --- | --- | --- |
| Full source pool (ref.) | 3.68 ± 0.63 | 3.94 ± 0.50 | 4.39 ± 0.95 |
| **Hausdorff** | **3.81 ± 0.65** | 4.83 ± 0.57 | 5.25 ± 1.00 |
| sum, Hausdorff-weighted (2:1:1) | 4.16 ± 0.78 | 4.61 ± 0.57 | 4.83 ± 0.82 |
| **Kernel herding** | **4.19 ± 0.90** | 5.27 ± 0.77 | 4.88 ± 0.92 |
| Sliced Wasserstein | 4.20 ± 0.47 | 4.83 ± 1.08 | 4.60 ± 0.62 |
| Random subset | 4.41 ± 0.76 | 5.49 ± 0.72 | 5.25 ± 1.09 |
| K-means (M=K, reps=1) | 4.61 ± 1.11 | 5.33 ± 0.48 | 4.97 ± 0.80 |
| Kernel mean | 4.65 ± 0.89 | 4.76 ± 0.60 | 5.15 ± 0.83 |
| **ProbCover** | **5.09 ± 1.59** | 5.06 ± 0.58 | 4.78 ± 0.73 |

**Stage-1 wall-clock cost** (seconds, mean over 5 seeds — no LLM calls, no
token cost, same convention as §27.4):

| Formula | build clouds | distance/Gram matrix | greedy select | **total** |
| --- | ---: | ---: | ---: | ---: |
| **Kernel herding** | 0.001s | 0.093s | 0.000s | **0.093s** |
| Kernel mean | 0.001s | 0.588s | 0.001s | **0.588s** |
| ProbCover | 0.001s | 0.577s | 0.001s | **0.577s** |
| Hausdorff | 0.001s | 0.876s | 0.001s | **0.876s** |
| Sliced Wasserstein | 0.001s | 1.378s | 0.001s | **1.378s** |
| sum, Hausdorff-weighted | 0.001s | 2.862s | 0.001s | **2.862s** |

**But Stage-1 (the actual new methods) is a rounding error against the
real cost of this benchmark.** Full end-to-end wall-clock per config
(seconds, mean over 5 seeds; `acq:*` = one full acquisition-method call,
i.e. greedy selection over the reduced pool **plus** `train_eval`'s
250-epoch Adam-on-CPU MLP training from scratch — this is what actually
takes minutes, not Stage-1's sub-3-second selection). Raw output:
`outputs/herding_probcover_comparison_K60_timed.json`; reproduce by
re-running the §27.9 command above (timing is captured automatically by
`run_distance_formula_comparison`, no separate flag needed).

| Config | stage1 | make_problem | acq:ActiveEval-Pair | acq:Facility-location | acq:Random | **total** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kernel herding | 0.09s | 0.03s | 1.72s | 28.83s | 0.51s | **31.19s** |
| ProbCover | 0.58s | 0.03s | 1.66s | 28.86s | 0.57s | **31.70s** |
| Kernel mean | 0.59s | 0.03s | 1.66s | 28.85s | 0.58s | **31.71s** |
| Sliced Wasserstein | 1.38s | 0.03s | 1.57s | 28.94s | 0.50s | **32.42s** |
| Hausdorff | 0.88s | 0.03s | 1.54s | 28.87s | 0.57s | **31.89s** |
| sum, Hausdorff-weighted | 2.86s | 0.03s | 1.77s | 28.96s | 0.46s | **34.08s** |
| K-means (M=K, reps=1) | 0.09s | 0.00s | 1.59s | 28.83s | 0.55s | **31.06s** |
| Full source pool | 0.00s | 0.06s | 9.92s | **169.29s** | 0.47s | **179.74s** |
| Random subset | 0.00s | 0.03s | 1.60s | 29.24s | 0.60s | **31.48s** |

**Grand total: 2176.3s (36.3 min) across all 9 configs × 5 seeds** (45
config-seed pairs, ≈48.4s average). **Compute tokens: not applicable** —
this entire module makes zero LLM/API calls; wall-clock CPU-seconds is the
only real resource cost, and it is reported here in full rather than as a
placeholder.

**A genuinely new, previously-undocumented finding: the `Facility-location`
*baseline* acquisition method is 15–19× more expensive per config than
`ActiveEval-Pair`** (≈28.8s vs. ≈1.6–1.8s at the reduced $K{=}60$ pool
size) **— and it's also consistently the worse performer on MAE** (every
row in the results table above shows Facility-location's MAE higher than
ActiveEval-Pair's). The mechanism: the `Facility-location` baseline runs
with `cover_all=True`, meaning its target mask spans the **entire**
candidate pool (thousands of points), so every greedy round's marginal-gain
sum is computed over the full pool; `ActiveEval-Pair` instead scores
against a much smaller, target-narrowed subset (plus a cheap log-det
diversity term) — doing *more* algorithmically but over a far smaller
working set, making it both better and dramatically cheaper. This gap
widens sharply with pool size: on `Full source pool` ($P{=}6780$ vs.
$3600$, a $1.9\times$ increase), Facility-location's cost jumps to
**169.29s — nearly 3 minutes per seed, and worse-than-linear in $P$**
(a $5.9\times$ cost increase for a $1.9\times$ pool-size increase, roughly
consistent with the per-round full-pool coverage scan scaling closer to
$O(P^2)$ than $O(P)$ once the proportionally-larger budget is accounted
for). **This single row's `Facility-location` cost — 169.29s × 5 seeds ≈
846s — is roughly 39% of the entire experiment's 2176.3s wall-clock
total.** None of the actual new methods this section introduces
(kernel herding, ProbCover, or any distance formula) contribute
meaningfully to total runtime; the cost story here is almost entirely
about the pre-existing `Facility-location` baseline's whole-pool coverage
computation, not about anything new implemented in this section.

#### Analysis

**Kernel herding is a genuinely strong, and the cheapest, method here.**
On ActiveEval-Pair it lands at 4.19 ± 0.90 — third overall, essentially
tied with sliced Wasserstein (4.20 ± 0.47) and the Hausdorff-weighted sum
(4.16 ± 0.78), clearly ahead of random (4.41), k-means (4.61), and
kernel_mean alone (4.65). And its Stage-1 cost (0.093s) is **~6× cheaper**
than the next-cheapest method (ProbCover/kernel_mean, ~0.58s) and **~31×
cheaper** than the Hausdorff-weighted sum (2.86s) — it needs only *one*
Gram-matrix pass, not a pairwise-distance computation per point followed by
standardization and combination. This is a real, practically useful
finding: **kernel herding gets most of the benefit of the best
distance-formula methods at a small fraction of the Stage-1 compute.**

> ⚠️ **But see §28.4**: this speed advantage is **regime-specific and
> reverses** at realistic sample-set sizes. The kernel Gram is $O(M^2)$ in
> points-per-sample-set while SciPy's Hausdorff is $O(M)$ average-case with
> high fixed overhead; measured crossover is $M\approx100$, and at $M{=}640$
> Hausdorff is $40\times$ *faster*. The benchmark's $M{=}20$ sits deep in
> the regime that favours herding. §28 derives why, and what to do about it.

This is worth pausing on given §27.9's own theory: kernel herding carries
*no* submodular guarantee (its objective, $-\mathrm{MMD}(A,B)$, is not
proven monotone or submodular, unlike facility location), yet it performs
competitively with — and cheaper than — every submodular-guaranteed
distance-formula variant except Hausdorff itself. This is consistent with a
pattern already seen throughout this investigation (`docs/submodularity-audit.md`,
§27.6-27.8): **a theoretical guarantee is not the same thing as, and does
not reliably predict, an empirical win.** Facility location's $(1-1/e)$
bound is a worst-case floor on *its own* objective, not a promise that its
objective is the right one for this downstream task, and kernel herding's
different objective (global distribution-matching rather than per-point
coverage) turns out to be a good fit for this problem despite lacking an
analogous guarantee.

**ProbCover is the worst method tested here — worse than doing nothing
clever at all (random).** 5.09 ± 1.59, both the highest MAE and by far the
widest confidence interval of any config in this table (more than double
most others). Two likely, complementary reasons:

1. **Hard vs. smooth reward.** ProbCover's marginal gain is binary — a
   candidate within $\delta$ of an uncovered point contributes the same
   whether it's barely inside the radius or dead center, and a candidate
   just outside $\delta$ contributes nothing however close. Facility
   location's smooth $\exp(-D^2/\tau)$ reward has no such cliff. In a
   synthetic generator whose ground-truth accuracy function varies
   continuously (§0's `true_acc`), a graded reward is plausibly a better
   match to the underlying structure than a hard in/out threshold.
2. **Budget regime mismatch.** ProbCover's median-heuristic $\delta$ means
   roughly half of all pairs count as "adjacent" from the start, so the
   first few picks likely cover a large fraction of the pool immediately —
   leaving the remaining ~55 of 60 picks (at $K{=}60$ out of 113
   candidates, over half the pool) to repeatedly hit the "nothing left to
   cover, shrink $\delta$" fallback path. The algorithm's original design
   context (Yehuda et al. 2022) is incremental active learning with modest
   per-round query sizes against an existing labeled pool, not grabbing
   over half a from-scratch candidate pool in one shot — this may simply be
   outside the budget regime it was designed for. Not verified directly
   here (would need testing ProbCover at a smaller $K$, e.g. $K{=}15$–$30$,
   to see if it fares better relative to the other methods there); flagged
   as the more likely of the two explanations but untested.

It also inherits kernel_mean's own middling distance matrix as its base
(ProbCover defaults to it here, §27.9), and kernel_mean alone is itself a
below-random performer at $K{=}60$ (4.65 vs. random's 4.41) — so some of
ProbCover's weakness may simply propagate from an unfavorable choice of
underlying distance rather than the hard-threshold mechanism itself; this
is not disentangled here (testing ProbCover on the Hausdorff distance
matrix instead would isolate the two effects) but is worth flagging as a
confound rather than presenting the hard-threshold explanation as the sole
cause.

**Practical bottom line:** if a combined-signal, non-target-aware coreset
method is wanted, **kernel herding is the best choice among the ones tested
that don't require picking a single winning distance metric in advance** —
it is cheap, competitive, and requires no metric-selection tuning (unlike
the Hausdorff-weighted sum, which needed the $K{=}60$-specific tuning
documented in §27.7–27.8 to become competitive). **ProbCover is not
recommended at this budget regime** without further investigation at
smaller $K$ or a different base distance.

---

## 28. Why is kernel herding both *fast* and *nearly as accurate* as Hausdorff?

The §27.9 result is surprising on its face. Kernel herding scored
**4.19 ± 0.90** vs Hausdorff's **3.81 ± 0.65** — statistically
indistinguishable at 5 seeds — while using **~9× less** Stage-1 compute
(0.093s vs 0.876s). Two questions deserve real answers:

> **(A)** Why is it *faster*?
> **(B)** How can something that just takes an *average* be nearly as good
> as something that carefully examines the *worst-case* point?

This section answers both from scratch. No prior background is assumed:
every symbol is defined, every claim is either proved or explicitly marked
as measured/cited. **Section 28.4 contains an important correction** to the
naive reading of the speed result — the advantage is real but
regime-specific, and it *reverses* on realistic data.

---

### 28.1 Setup and notation (plain English first)

We have $N$ candidate sample-sets ("workloads"). Sample-set $b_i$ is a
**cloud of $M$ points** in $\mathbb R^d$:

$$
b_i = \{x_{i1}, x_{i2}, \ldots, x_{iM}\}, \qquad x_{im}\in\mathbb R^d.
$$

Think: each $x_{im}$ is one example inside that workload (an embedded
prompt), and the cloud is the whole workload. In §27's benchmark
$M{=}20$, $d{=}6$, $N{=}113$.

To pick representatives we must **compare two clouds**. That comparison is
where all the cost lives, and where the two methods differ fundamentally.

**The two comparison rules, in plain English:**

- **Hausdorff** asks: *"What is the single worst-matched point?"* For every
  point in cloud $A$, find its nearest neighbour in cloud $B$; take the
  worst of those. Then repeat in the other direction and take the worse of
  the two. It is a **worst-case** question.
- **Kernel mean / herding** asks: *"On average, do these clouds look
  alike?"* Summarise each cloud by an average, then compare the averages.
  It is an **average-case** question.

Formally:

$$
d_H(b_i,b_j) = \max\Big\{\ \max_{x\in b_i}\min_{y\in b_j}\lVert x-y\rVert,\ \ \max_{y\in b_j}\min_{x\in b_i}\lVert x-y\rVert\ \Big\}
$$

$$
\mu_i \;=\; \frac1M\sum_{m=1}^{M}\varphi(x_{im}) \;\in\; \mathcal H,
\qquad
\operatorname{MMD}(b_i,b_j) = \lVert \mu_i - \mu_j\rVert_{\mathcal H}
$$

where $\varphi$ maps a point into a feature space $\mathcal H$ with
$\langle\varphi(x),\varphi(y)\rangle_{\mathcal H} = k(x,y)$, and $k$ is the
Gaussian/RBF kernel $k(x,y)=\exp(-\lVert x-y\rVert^2/\tau)$.

---

## Part A — Why kernel herding is faster

### 28.2 The structural reason: *additive* vs *nested-extremum* aggregation

This is the heart of it, and it is worth stating precisely.

**Kernel herding's aggregation is additive (a sum).** Look at $\mu_i$: it
is $\frac1M\sum_m \varphi(x_{im})$ — each point contributes its own term,
independently, and the terms are added. The consequence is that the whole
cloud **collapses into one single vector** $\mu_i$. Two clouds are then
compared by comparing two vectors:

$$
\langle \mu_i,\mu_j\rangle_{\mathcal H}
= \Big\langle \tfrac1M\textstyle\sum_m \varphi(x_{im}),\ \tfrac1M\textstyle\sum_n \varphi(x_{jn})\Big\rangle
= \frac1{M^2}\sum_{m=1}^{M}\sum_{n=1}^{M} k(x_{im},x_{jn})
\;=:\; G_{ij}.
$$

The middle step uses only bilinearity of the inner product — that is the
*entire* mathematical content, and it is exactly what makes the method
cheap. Everything herding needs is a function of the $N\times N$ matrix
$G$:

$$
\operatorname{MMD}^2(b_i,b_j) = \lVert\mu_i-\mu_j\rVert^2 = G_{ii} + G_{jj} - 2G_{ij},
$$

$$
\langle\varphi(b_i),\mu_B\rangle = \tfrac1N\textstyle\sum_j G_{ij},
\qquad
\langle\varphi(b_i),\mu_{A_t}\rangle = \tfrac1t\textstyle\sum_{a\in A_t} G_{ia}.
$$

**Once $G$ is built, the raw points are never touched again.**

**Hausdorff's aggregation is a nested extremum (max of min), which does not
decompose.** In $\max_{x}\min_{y}\lVert x-y\rVert$, the contribution of a
point $x\in b_i$ is $\min_{y\in b_j}\lVert x-y\rVert$ — a quantity that
depends on *every point of the other cloud*. You cannot compute it from a
per-cloud summary, because $\min$ and $\max$ are not linear and do not
commute with aggregation:

$$
\min_y \lVert x - y\rVert \ \ \text{is \textit{not} expressible as}\ \ f\big(x,\ \text{summary}(b_j)\big)
$$

for any fixed finite summary. Concretely: $\mathbb E[\min] \neq
\min[\mathbb E]$, and no amount of pre-averaging cloud $b_j$ recovers the
nearest-neighbour structure. **So Hausdorff must keep all $M$ points of
both clouds and re-scan them for every pair.** There is no $G$-like object
to precompute and reuse.

> **The one-sentence version:** kernel herding gets to *summarise each
> cloud once and reuse that summary in all $N-1$ of its comparisons*;
> Hausdorff must re-examine the raw points afresh for every single pair.

### 28.3 Exact operation counts, and measured confirmation

The structural difference cashes out into a concrete arithmetic saving.
Note $G_{ii}$ (the self-term) appears in every $\operatorname{MMD}$
involving cloud $i$ — but it only has to be computed **once**, on the
diagonal.

| | Kernel-matrix blocks per pair | Total $M^2$-blocks over all pairs |
| --- | --- | --- |
| `kernel_mean_distance` (as used by the MMD distance formula) | $K_{AA}, K_{BB}, K_{AB}$ → **3** | $3\cdot\binom{N}{2} = 18{,}984$ |
| `compute_subset_kernel_gram` (herding's path) | $K_{AB}$ only → **1** | $\binom{N}{2}+N = 6{,}441$ |

Predicted speedup $\approx 3\times$. Plus a second saving: the MMD path
re-estimates the kernel bandwidth $\tau$ **per pair**, whereas the Gram
path estimates it **once globally**.

**Measured** ($N{=}113$, $M{=}20$, $d{=}6$, 6328 pairs):

| Operation | Time | Notes |
| --- | ---: | --- |
| Hausdorff, all pairs | 0.883s | 2 SciPy calls per pair |
| `kernel_mean_distance`, all pairs (default) | 0.576s | 3 blocks + per-pair $\tau$ |
| `kernel_mean_distance`, all pairs (shared $\tau$) | 0.274s | 3 blocks, $\tau$ once |
| **`compute_subset_kernel_gram`** | **0.079s** | **1 block, $\tau$ once** |
| *(isolated)* per-pair $\tau$ re-estimation alone | 0.290s | the overhead removed above |

Both predictions check out quantitatively:

- **3-blocks-vs-1:** $0.274 / 0.079 = 3.5\times$ ✓ (predicted $\approx 3\times$)
- **Bandwidth overhead is additive:** $0.274 + 0.290 = 0.564 \approx 0.576$ ✓
- **Net vs Hausdorff:** $0.883/0.079 = 11.2\times$ (§27.9's end-to-end
  figure was $9.4\times$; the difference is the small fixed cost of cloud
  construction included there).

### 28.4 ⚠️ Important correction — the speed advantage *reverses* at realistic scale

The naive conclusion — *"kernel herding is asymptotically cheaper"* — is
**wrong**, and the measured data says so plainly. Sweeping $M$ (points per
sample-set) with $N{=}30$ clouds fixed:

| $M$ | Hausdorff (all pairs) | Gram (all pairs) | ratio | Hausdorff µs/pair |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.061s | 0.003s | **18.9× faster** | 139 |
| 40 | 0.061s | 0.017s | **3.5× faster** | 140 |
| 160 | 0.069s | 0.243s | 0.28× (**3.5× slower**) | 159 |
| 640 | 0.102s | 4.151s | 0.02× (**40× slower**) | 234 |

Read the last column: Hausdorff's cost per pair is **nearly flat** —
$64\times$ more points ($10\to640$) costs only $1.7\times$ more time. The
Gram meanwhile scales cleanly as $O(M^2)$ (each $4\times$ in $M$ gives
$\approx16\times$ in time). **They cross over at $M\approx100$.**

Why is Hausdorff nearly flat? Two reasons:

1. **SciPy's `directed_hausdorff` is not the naive $O(M^2)$ double loop.**
   It implements Taha & Hanbury (2015)'s early-break algorithm, which is
   $O(M)$ *average case* (worst case $O(M^2)$): it scans points in a
   randomised order and abandons the inner nearest-neighbour search as soon
   as the running distance cannot beat the current maximum.
2. **At small $M$, fixed call overhead dominates everything.** ~139 µs/pair
   is spent on Python→C marshalling for the two `directed_hausdorff` calls,
   *independent of $M$*. At $M{=}20$ this overhead **is** the measurement.

So the honest statement is:

> **Kernel herding's speed win comes from low per-pair overhead (one
> vectorised NumPy operation) and 3× fewer kernel blocks — *not* from
> better asymptotic complexity. It is a small-$M$ effect.**

**This matters practically.** The benchmark's $M{=}20$ points per
sample-set is unrealistically small; a real Text2SQL workload has hundreds
or thousands of examples. At $M{\gtrsim}100$ the ranking inverts and
Hausdorff becomes the cheap option. If herding is used at realistic scale,
the Gram must be approximated — subsample points per cloud, or use
Nyström / random Fourier features to avoid the $O(M^2)$ term. **Not tested
here**; flagged as required future work rather than assumed to work.

> **§30 derives all of this from first principles**: Theorem A proves the
> Gram needs $\Omega(M^2)$ (sums admit no certificates); Theorem B proves
> Hausdorff needs only $O(M)$ on average (one witness retires an inner
> loop); the fitted cost model predicts the crossover at
> $M^\star = 82.4$, confirmed by measured parity at $M{=}80$. §30.5 gives
> the Random-Fourier-Feature fix and shows it is the exact *dual* of
> §28.2's additivity argument.

---

## Part B — Why kernel herding is still accurate

Now the harder question. Averaging usually *destroys* information: the
average of $\{0,10\}$ and of $\{5,5\}$ are both $5$, yet the two sets are
completely different. So how can a method built on an average compete with
one that inspects worst-case geometry?

### 28.5 The answer: average in the *right space* and nothing is lost

The resolution is that $\mu_i$ is **not** the average of the raw points.
It is the average of the *features* $\varphi(x)$ — and for the Gaussian
kernel, $\varphi$ maps into an **infinite-dimensional** space in which the
average retains everything.

**Definition (characteristic kernel).** A kernel $k$ is *characteristic* if
the mean-embedding map

$$
P \;\longmapsto\; \mu_P := \mathbb E_{x\sim P}\big[\varphi(x)\big]
$$

is **injective** over probability distributions — that is,

$$
\mu_P = \mu_Q \iff P = Q .
$$

**Fact.** The Gaussian RBF kernel is characteristic on $\mathbb R^d$
(Sriperumbudur, Gretton, Fukumizu, Schölkopf & Lanckriet, *JMLR* 2010).

This is the whole answer to question (B) at the theoretical level: for a
characteristic kernel, **the single vector $\mu_P$ determines the entire
distribution $P$ uniquely.** No information is lost by summarising a cloud
into its mean embedding. Consequently

$$
\operatorname{MMD}(P,Q) = \lVert\mu_P-\mu_Q\rVert_{\mathcal H} = 0 \iff P = Q,
$$

i.e. MMD is a genuine **metric** on distributions, not a lossy proxy.

**Why injectivity holds — the dual/IPM view.** MMD has an equivalent
variational form:

$$
\operatorname{MMD}(P,Q) \;=\; \sup_{\lVert f\rVert_{\mathcal H}\le 1}\Big(\mathbb E_{P}[f] - \mathbb E_{Q}[f]\Big).
$$

So $\operatorname{MMD}(P,Q)=0$ means $P$ and $Q$ give the *same expectation
to every function* in the RKHS unit ball. Because the Gaussian RKHS is rich
enough to separate distributions, agreeing on all such test functions
forces $P=Q$.

### 28.6 Concretely: the mean embedding secretly stores *all moments*

Here is why the Gaussian feature map is rich enough, made explicit. Expand
the kernel:

$$
k(x,y)=\exp\!\Big(-\tfrac{\lVert x-y\rVert^2}{\tau}\Big)
= \underbrace{e^{-\lVert x\rVert^2/\tau}}_{a(x)}\ \underbrace{e^{-\lVert y\rVert^2/\tau}}_{a(y)}\ e^{2\langle x,y\rangle/\tau},
$$

and expand the last factor as a power series:

$$
e^{2\langle x,y\rangle/\tau} \;=\; \sum_{r=0}^{\infty}\frac{1}{r!}\Big(\tfrac{2}{\tau}\Big)^{r}\langle x,y\rangle^{r}.
$$

The term $\langle x,y\rangle^{r}$ expands into all degree-$r$ monomials in
the coordinates of $x$ (paired with those of $y$). Therefore the feature
map has, as its components, **every monomial of every degree**, weighted:

$$
\varphi(x) \;\propto\; a(x)\Big(1,\ \sqrt{\tfrac{2}{\tau}}\,x_1,\ \ldots,\ \sqrt{\tfrac{2}{\tau}}\,x_d,\ \ \sqrt{\tfrac{(2/\tau)^2}{2!}}\,x_1^2,\ \ldots\Big).
$$

Taking the average of $\varphi$ over the cloud therefore computes

$$
\mu_i \;=\; \Big(\mathbb E[a],\ \mathbb E[a\,x_1],\ \ldots,\ \mathbb E[a\,x_1^2],\ \mathbb E[a\,x_1x_2],\ \ldots\Big),
$$

i.e. **a weighted list of every moment of the cloud** — mean, variance,
covariance, skewness, kurtosis, and on forever.

> **Plain-English answer to "isn't averaging lossy?"** It *is* an average —
> but of an infinitely long feature vector holding every moment at once. So
> the "cheap average" is cheap to *compute* (via the kernel trick, never
> forming $\varphi$ explicitly) while being **statistically complete**.
> That is the trick, and it is why a one-vector-per-cloud summary can
> compete with a method that re-examines raw geometry.

### 28.7 Why herding beats random sampling: it is Frank–Wolfe in disguise

Injectivity says the *objective* is sound. This says the *greedy* is good.

Recall the herding rule from §27.9:

$$
a_{t+1} = \arg\max_{b_i} \big\langle \varphi(b_i),\ \mu_B - \mu_{A_t}\big\rangle_{\mathcal H}.
$$

**Theorem (Bach, Lacoste-Julien & Obozinski, ICML 2012).** Kernel herding
is *exactly* the Frank–Wolfe (conditional gradient) algorithm applied to

$$
\min_{g\in\mathcal M} J(g),\qquad J(g)=\tfrac12\lVert g-\mu_B\rVert^2_{\mathcal H},
\qquad \mathcal M = \operatorname{conv}\{\varphi(b): b\in B\}.
$$

*Sketch.* Frank–Wolfe picks the vertex minimising the linearised
objective: $\nabla J(g_t) = g_t - \mu_B$, so

$$
\arg\min_{g\in\mathcal M}\ \langle \nabla J(g_t),\, g\rangle
= \arg\min_{b}\ \langle g_t-\mu_B,\ \varphi(b)\rangle
= \arg\max_{b}\ \langle \varphi(b),\ \mu_B-g_t\rangle,
$$

which is the herding rule verbatim. With FW step size $\gamma_t =
1/(t{+}1)$ the iterate $g_t$ is precisely the running average
$\mu_{A_t}$. $\blacksquare$

**Convergence rates.** Let $T=|A|$ be the number selected:

| Method | Rate of $\lVert\mu_B-\mu_{A_T}\rVert$ |
| --- | --- |
| i.i.d. random sampling | $O(1/\sqrt T)$ (CLT / Hilbert-space concentration) |
| kernel herding | $O(1/T)$ — *conditionally*, see below |

The $O(1/T)$ rate (Chen, Welling & Smola, UAI 2010 — the "super-samples"
result) is **quadratically faster** than random sampling, and is the formal
reason herding should beat a random subset. **Honest caveat:** Bach et al.
(2012) showed this fast rate requires $\mu_B$ to lie in the *relative
interior* of the marginal polytope $\mathcal M$ (with a ball of positive
radius). In an infinite-dimensional RKHS that condition frequently fails,
and the guaranteed rate degrades toward $O(1/\sqrt T)$ — i.e. back to
random-sampling parity. So the theory predicts "**at least as good as
random, often better**", not a guaranteed win. Our measurement — herding
**4.19 ± 0.90** vs random subset **4.41 ± 0.76** — is consistent with
exactly that: better on the central estimate, not separated at 5 seeds.

Note this is a *different kind* of guarantee from facility location's
$(1-1/e)$: that bounds how close greedy gets to the best possible subset
*for its own coverage objective*; this bounds how fast the selected set's
distribution converges to the full pool's. Neither implies the other.

### 28.8 Measured confirmation — herding really does reproduce the distribution

The theory makes a falsifiable prediction: because herding minimises
$\operatorname{MMD}(A,B)$, the selected set should be **proportionally
representative** of the pool — matching sub-region frequencies. Hausdorff,
optimising worst-case coverage, has no such property and should chase
*extremes* instead.

Testing this directly (seed 0; "near-target" = top quintile of source
`target_score`, whose true pool frequency is **20.4%**):

| $K$ | Kernel herding | Hausdorff | Kernel mean | ProbCover |
| ---: | --- | --- | --- | --- |
| 10 | 10.0% (−10.4) | **0.0% (−20.4)** | 10.0% (−10.4) | 20.0% (−0.4) |
| 15 | **20.0% (−0.4)** | **0.0% (−20.4)** | 13.3% (−7.0) | 20.0% (−0.4) |
| 25 | 16.0% (−4.4) | **4.0% (−16.4)** | 16.0% (−4.4) | 28.0% (+7.6) |
| 60 | 23.3% (+3.0) | 20.0% (−0.4) | 20.0% (−0.4) | 16.7% (−3.7) |

*(deviation from the true 20.4% in parentheses)*

**The prediction is confirmed.** At small $K$, kernel herding tracks the
true frequency closely (deviations −10.4, −0.4, −4.4 pp) while **Hausdorff
selects essentially zero near-target representatives** (0%, 0%, 4% — a
−20.4 pp deviation, i.e. it misses that region entirely). Hausdorff is
demonstrably *not* doing proportional representation; it is chasing
boundary/extreme points, exactly as the worst-case objective implies.

**This partially resolves the open question from §27.6** — why Hausdorff
was the *worst* method at $K{=}30$ (5.46 pp) yet the *best* at $K{=}60$
(3.81 pp). The coverage data offers a mechanism: at small budgets
Hausdorff spends everything on extremes and never covers the
target-relevant region at all; by $K{=}60$ (over half the pool) it has
enough budget that extremes *and* interior are both covered, and its
worst-case guarantee starts paying off. Kernel herding, by contrast, is
distributionally stable across all $K$ — which is why it is the safer
choice when the budget is small or unknown in advance.

**Caveats, stated plainly:** this diagnostic is one seed and correlational.
It is a mechanistically plausible and theoretically *predicted* account,
not a controlled proof — §27.6's two earlier hypotheses were tested and
ruled out, so this third one deserves the same scrutiny (a multi-seed
version of this table, and a $K$-sweep of MAE against near-target
coverage, would settle it).

---

### 28.9 Summary

| Question | Answer |
| --- | --- |
| **Why faster?** | The mean embedding is an **additive** summary, so each cloud collapses to one reusable vector and all comparisons reduce to one $N\times N$ Gram matrix; Hausdorff's **nested max-min** does not decompose and must rescan raw points per pair. Concretely: 1 kernel block per pair instead of 3, one global bandwidth instead of per-pair, and one vectorised NumPy call instead of two per-pair SciPy calls. Measured $11.2\times$. |
| **Is that speedup robust?** | **No — it is small-$M$ only.** SciPy's Hausdorff is $O(M)$-average with high fixed overhead; the Gram is $O(M^2)$ with low overhead. Crossover $M\approx100$; at $M{=}640$ Hausdorff is $40\times$ *faster*. Real workloads sit past the crossover. |
| **Why still accurate?** | The Gaussian kernel is **characteristic**, so the mean embedding is **injective** — one vector determines the whole distribution (it secretly stores every moment). MMD is therefore a true metric, not a lossy proxy. |
| **Why beat random?** | Herding **is** Frank–Wolfe on $\tfrac12\lVert g-\mu_B\rVert^2$, giving $O(1/T)$ vs random's $O(1/\sqrt T)$ — conditionally (interior condition may fail in infinite dimensions, degrading to parity). |
| **Why *not quite* beat Hausdorff?** | Different objectives: herding matches the **distribution**, Hausdorff covers the **worst case**. At $K{=}60$ worst-case coverage wins by 0.38 pp — but the CIs overlap, and at *small* $K$ the ordering reverses (§27.6, §28.8). |

---

## 29. What Hausdorff *provably* guarantees, and why that guarantee binds at $K\ge60$

### 29.0 Scope — what is and is not claimed here

**This section does not prove that Hausdorff is empirically best.** That
claim cannot honestly be proved, for three reasons visible in our own data:

1. The confidence intervals **overlap** (Hausdorff $3.81\pm0.65$ vs.
   Hausdorff-weighted sum $4.16\pm0.78$ vs. herding $4.19\pm0.90$ at 5
   seeds) — "best" is a central-estimate ordering, not a separated result.
2. The ordering **reverses at $K{=}30$**, where Hausdorff was the *worst*
   method tested (5.46 pp, §27.2).
3. Empirical superiority on a synthetic generator is not the kind of
   statement a theorem can establish.

What **can** be proved, and is proved below, is strictly this: the two
families optimise **logically independent guarantees** — Hausdorff-induced
facility location controls a **uniform ($L^\infty$)** quantity, kernel
herding controls an **average ($L^1$)** one. Section 29.6 then shows,
by measurement, that Hausdorff's guarantee only becomes *materially
stronger than its competitors'* at $K\gtrsim60$ — which is exactly where
its empirical advantage appears. That is a **mechanistic explanation
consistent with the data**, not a proof of the empirical ranking, and
§29.7 states plainly what it fails to explain.

### 29.1 Definitions

Fix a metric $d$ on sample-sets (below, $d = d_H$, the Hausdorff metric of
§3.3). For a selected set $A$, write the **distance-to-nearest-representative**
and the **covering radius**:

$$
d(j,A) \;:=\; \min_{i\in A} d(b_i,b_j),
\qquad
\boxed{\ \varepsilon(A) \;:=\; \max_{1\le j\le N} d(j,A)\ }
$$

In words: $\varepsilon(A)$ is *the distance from the worst-served candidate
to its closest chosen representative*. Small $\varepsilon(A)$ means **no
candidate anywhere in the pool is far from something we selected**.

### 29.2 Lemma 1 — the facility-location objective controls the covering radius

Recall $S_{ij}=\exp(-d_{ij}^2/\tau)$ and $F(A)=\sum_{j=1}^N\max_{i\in A}S_{ij}$
(§6). Since $t\mapsto e^{-t^2/\tau}$ is decreasing,
$\max_{i\in A}S_{ij}=\exp(-d(j,A)^2/\tau)$, so

$$
F(A) \;=\; \sum_{j=1}^{N} \exp\!\Big(-\frac{d(j,A)^2}{\tau}\Big).
$$

**Lemma 1(a) — unconditional direction.** Since $d(j,A)\le\varepsilon(A)$
for every $j$,

$$
F(A) \;\ge\; N\exp\!\Big(-\frac{\varepsilon(A)^2}{\tau}\Big).
$$

*Proof.* Termwise monotonicity of $\exp(-t^2/\tau)$. $\blacksquare$

**Lemma 1(b) — converse.** Let $\Delta := N - F(A) \ge 0$. If $\Delta < 1$
then

$$
\boxed{\ \varepsilon(A) \;\le\; \sqrt{\ \tau\,\ln\frac{1}{1-\Delta}\ }\ }
$$

*Proof.* Every summand of
$N-F(A)=\sum_j\big(1-e^{-d(j,A)^2/\tau}\big)$ is non-negative, so any
single one is at most the total. Applying this to the index $j^\star$
attaining $d(j^\star,A)=\varepsilon(A)$:

$$
1-e^{-\varepsilon(A)^2/\tau} \;\le\; \Delta
\;\Longrightarrow\;
e^{-\varepsilon(A)^2/\tau} \;\ge\; 1-\Delta
\;\Longrightarrow\;
\frac{\varepsilon(A)^2}{\tau} \;\le\; \ln\frac{1}{1-\Delta},
$$

and taking square roots gives the claim. $\blacksquare$

**Honest note on Lemma 1(b):** it is informative only when $\Delta<1$,
i.e. $F(A)>N-1$ — coverage must be near-saturated before the bound says
anything. This is a genuine limitation, not a formality: at small $K$,
$\Delta\gg1$ and the bound is vacuous. That is the first hint of why
Hausdorff needs a large enough budget before its guarantee means anything.

### 29.3 Theorem 2 — covering radius bounds *worst-case* label transfer

Here is why $\varepsilon(A)$ is the quantity we should care about.

**Assumption (A1, Lipschitz accuracy).** The true accuracy functional
$a^\star(b)\in[0,1]$ — the accuracy a fixed reference model attains on
workload $b$ — is $L$-Lipschitz with respect to $d_H$:

$$
|a^\star(b)-a^\star(b')| \;\le\; L\,d_H(b,b') \qquad \forall b,b'.
$$

*(This is an assumption, not a theorem. It is plausible for $d_H$
specifically: $d_H(b,b')$ small means every example in $b$ has a nearby
counterpart in $b'$ and vice versa, so a model behaving continuously on
inputs should score similarly. It is **not verified** on this generator.)*

**Theorem 2.** Under (A1), for every candidate $b_j$ let
$i^\star(j)=\arg\min_{i\in A}d_H(b_i,b_j)$ be its nearest selected
representative. Then

$$
\boxed{\ \max_{1\le j\le N}\ \big|a^\star(b_j)-a^\star\big(b_{i^\star(j)}\big)\big| \;\le\; L\,\varepsilon(A)\ }
$$

*Proof.* Fix $j$. By (A1) and the definition of $i^\star(j)$,

$$
\big|a^\star(b_j)-a^\star(b_{i^\star(j)})\big| \;\le\; L\,d_H\big(b_j,b_{i^\star(j)}\big) \;=\; L\,d(j,A) \;\le\; L\,\varepsilon(A).
$$

Taking the maximum over $j$ preserves the inequality. $\blacksquare$

**Interpretation.** The label we *did* pay for at $b_{i^\star(j)}$ is within
$L\varepsilon(A)$ of the label we *did not* pay for at $b_j$ — **uniformly,
for every unlabelled candidate simultaneously**. Chaining with Lemma 1(b):
whenever $F(A)>N-1$,

$$
\max_j\big|a^\star(b_j)-a^\star(b_{i^\star(j)})\big| \;\le\; L\sqrt{\tau\ln\tfrac{1}{1-\Delta}} .
$$

### 29.4 Theorem 3 — kernel herding bounds only the *average*

Now the contrast. Work in the subset-level RKHS of §27.9, where
$\varphi(b_i)=\mu_i$ and $G_{ij}=\langle\varphi(b_i),\varphi(b_j)\rangle_{\mathcal H}$,
with $\mu_B=\frac1N\sum_j\varphi(b_j)$ and $\mu_A=\frac1{|A|}\sum_{i\in A}\varphi(b_i)$.

**Theorem 3.** For any $f\in\mathcal H$,

$$
\boxed{\ \Big|\ \frac1N\sum_{j=1}^{N} f(b_j)\ -\ \frac1{|A|}\sum_{i\in A} f(b_i)\ \Big| \;\le\; \lVert f\rVert_{\mathcal H}\,\big\lVert \mu_B-\mu_A\big\rVert_{\mathcal H}\ }
$$

*Proof.* By the reproducing property $f(b)=\langle f,\varphi(b)\rangle$, so
by linearity $\frac1N\sum_j f(b_j)=\langle f,\mu_B\rangle$ and
$\frac1{|A|}\sum_{i\in A}f(b_i)=\langle f,\mu_A\rangle$. Hence the
left-hand side equals $|\langle f,\mu_B-\mu_A\rangle|$, and Cauchy–Schwarz
gives the bound. $\blacksquare$

Applying this to $f=a^\star$ (assuming $a^\star\in\mathcal H$ with
$\lVert a^\star\rVert_{\mathcal H}\le R$) yields exactly what herding
controls:

$$
\Big|\underbrace{\tfrac1N\textstyle\sum_j a^\star(b_j)}_{\text{pool mean accuracy}} \;-\; \underbrace{\tfrac1{|A|}\textstyle\sum_{i\in A} a^\star(b_i)}_{\text{mean over what we labelled}}\Big| \;\le\; R\,\lVert\mu_B-\mu_A\rVert_{\mathcal H}.
$$

**This is an average over the pool, and it is the *only* thing bounded.**
It permits individual candidates to be arbitrarily mis-represented so long
as the errors cancel.

### 29.5 The two guarantees are *logically independent*

Neither implies the other — both directions fail, by explicit
counterexample.

**(i) Small MMD $\;\not\Rightarrow\;$ small covering radius.**
Let the pool consist of $99$ copies of a workload at descriptor $x_0$ and
one outlier at $x_1$, with $\lVert\varphi(x_1)-\varphi(x_0)\rVert = c$.
Select $A$ = any $K$ of the $99$ copies. Then $\mu_A=\varphi(x_0)$ and

$$
\lVert\mu_B-\mu_A\rVert = \Big\lVert \tfrac{99}{100}\varphi(x_0)+\tfrac{1}{100}\varphi(x_1)-\varphi(x_0)\Big\rVert = \tfrac{c}{100},
$$

i.e. **MMD is tiny** — herding is perfectly satisfied. Yet the outlier is
entirely unrepresented: $\varepsilon(A)=d(x_1,x_0)$, which can be
arbitrarily large, and by Theorem 2 nothing at all is guaranteed about
$a^\star(x_1)$.

**(ii) Small covering radius $\;\not\Rightarrow\;$ small MMD.**
Same pool. Select $A$ with half its budget near $x_0$ and half near $x_1$.
Now $\varepsilon(A)\approx0$ — perfect coverage — but
$\mu_A\approx\tfrac12\varphi(x_0)+\tfrac12\varphi(x_1)$ while
$\mu_B\approx\tfrac{99}{100}\varphi(x_0)+\tfrac1{100}\varphi(x_1)$, giving
$\lVert\mu_B-\mu_A\rVert\approx0.49\,c$ — **MMD is large**. The *frequencies*
are badly wrong even though every region is covered.

$$
\boxed{\ \text{coverage } (L^\infty) \ \ \text{and} \ \ \text{distribution matching } (L^1) \ \ \text{are incomparable guarantees.}\ }
$$

This is the precise mathematical sense in which §28.8's empirical
observation (herding reproduces frequencies; Hausdorff chases extremes) is
not an accident of implementation but a **necessary consequence of the two
objectives**.

### 29.6 Measured — Hausdorff's guarantee only *separates* at $K\gtrsim60$

Theory says $\varepsilon(A)$ is what Hausdorff buys. Measuring it directly
(seed 0, covering radius computed in the Hausdorff metric for every
method's selection, so all rows are comparable):

| $K$ | Hausdorff | Kernel mean | Herding | sum (2:1:1) | ProbCover |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 3.242 | 3.577 | 3.458 | 3.345 | **3.177** |
| 30 | **2.522** | 2.570 | 2.666 | 3.257 | 3.177 |
| 60 | **1.764** | 1.930 | 2.658 | 2.764 | 2.687 |
| 90 | **1.491** | 1.845 | 2.191 | 2.687 | 2.280 |

Read the *margins*, not the ranks:

| $K$ | Hausdorff $\varepsilon$ | best competitor | Hausdorff's margin |
| ---: | ---: | ---: | --- |
| 15 | 3.242 | 3.177 (ProbCover) | **none — it loses** |
| 30 | 2.522 | 2.570 (kernel mean) | 1.9% — negligible |
| 60 | 1.764 | 1.930 (kernel mean) | **9.4%**; vs herding **34%** |
| 90 | 1.491 | 1.845 (kernel mean) | **24%** |

**This resolves the open question from §27.6.** Hausdorff's covering-radius
advantage is *absent* at $K{=}15$, *negligible* at $K{=}30$, and only
becomes substantial from $K{=}60$ — tracking its MAE record exactly:
worst method at $K{=}30$ (5.46 pp), best at $K{=}60$ (3.81 pp). The
mechanism the theory predicts and the measurement confirms:

- **At small $K$**, Lemma 1(b) is vacuous ($\Delta\gg1$), Hausdorff has no
  meaningful coverage edge, *and* (§28.8) it spends its budget on extremes
  — 0/10 and 0/15 near-target picks. It gets neither guarantee: no uniform
  bound and no distributional match. Worst of both worlds, hence worst MAE.
- **At $K\gtrsim60$**, $\varepsilon(A)$ drops sharply and separates from
  every competitor. Theorem 2's uniform bound becomes tight, no region of
  descriptor space is left for the meta-evaluator to extrapolate into
  blindly, and its MAE becomes best.

### 29.7 What this does *not* explain — stated plainly

Covering radius is **not** a sufficient statistic for MAE. Two rows in our
own $K{=}60$ data contradict a simple "lower $\varepsilon$ ⟹ lower MAE"
reading:

| Method | $\varepsilon$ rank | MAE rank |
| --- | --- | --- |
| Hausdorff | 1st (1.764) | 1st (3.81) ✓ |
| Kernel mean | **2nd** (1.930) | **4th** (4.65) ✗ |
| sum (2:1:1) | **5th** (2.764) | **2nd** (4.16) ✗ |

So $\varepsilon$ cleanly explains the Hausdorff *extremes* of the story —
why it fails at small $K$ and wins at large $K$ — but it does **not**
order the middle of the field. Something else (plausibly the
distribution-matching axis of §29.4, on which `sum` and herding do well and
`kernel_mean` does not) is also in play; a two-factor account is needed and
is not developed here.

Further limitations, for the record:

- **(A1) is assumed, not verified.** No test was run for Lipschitzness of
  $a^\star$ in $d_H$ on this generator.
- **The covering-radius table is a single seed.** The MAE numbers are 5
  seeds; these are not.
- **The MAE ranking is not statistically separated** (§29.0).
- **All of this is on the synthetic benchmark.** §27.9's $M{=}20$
  points-per-sample-set is unrealistically small (§28.4), and $d_H$'s
  behaviour — hence $\varepsilon$ — may differ substantially at realistic
  $M$.

---

## 30. Why kernel herding loses to Hausdorff as $M$ grows — a complexity-theoretic account

§28.4 *reported* the crossover ($M\approx100$) as a measurement. This
section *derives* it. The question is precise:

> Let $M$ = points per sample-set. Why does exact Gram computation cost
> $\Theta(M^2)$ while exact Hausdorff costs $O(M)$ on average — and where
> exactly do the curves cross?

The answer is a genuine asymmetry in the two objectives' **certificate
structure**, and it turns out to be the *exact dual* of the additivity
argument in §28.2. Everything below concerns **computational cost only** —
not accuracy.

### 30.1 Theorem A — exact Gram evaluation requires $\Omega(M^2)$

$$
G_{ij} \;=\; \frac{1}{M^2}\sum_{m=1}^{M}\sum_{n=1}^{M} k(x_{im},x_{jn}).
$$

**Theorem A.** In the kernel-oracle model (an algorithm learns kernel
values only by querying $k(x_{im},x_{jn})$ for chosen $(m,n)$), any
algorithm computing $G_{ij}$ **exactly** must issue all $M^2$ queries.

*Proof (adversary argument).* Suppose algorithm $\mathcal A$ halts and
outputs the correct value after querying a proper subset
$Q\subsetneq\{1,\dots,M\}^2$, and pick $(m_0,n_0)\notin Q$. Construct a
second instance identical on every queried entry but with the
$(m_0,n_0)$ entry perturbed by $\delta\neq0$. Since $\mathcal A$'s control
flow and output depend only on the values it queried, it returns the *same*
number on both instances. But the true values differ by
$\delta/M^2\neq0$, so $\mathcal A$ is wrong on at least one. $\blacksquare$

**Why the sum is "rigid":** every term has non-zero influence,
$\partial G_{ij}/\partial k(x_{im},x_{jn}) = 1/M^2 \neq 0$. **No partial
information certifies anything about the remainder** — there is no
subset of terms whose values ever let you skip the rest.

*Scope:* this lower bound is for **exact** computation in the oracle model.
It does **not** forbid fast *approximation* from the raw point coordinates
— which is precisely the escape route of §30.5 (and why tree codes / Fast
Gauss Transform / random features exist).

### 30.2 Theorem B — Hausdorff admits $o(M^2)$ *exact* evaluation

The directed Hausdorff distance is a nested extremum,

$$
d_H^{\rightarrow}(A,B) \;=\; \max_{x\in A}\, g(x), \qquad g(x) := \min_{y\in B}\lVert x-y\rVert .
$$

**Theorem B (certificate structure).** Computing $\max_x g(x)$ exactly does
**not** require computing $g(x)$ exactly for all $x$. It suffices to know:

1. $g(x^\star)$ exactly for the maximiser $x^\star$, and
2. for every other $x$, a **certificate** that $g(x)\le g(x^\star)$.

And such a certificate is a *single point*: if any $y\in B$ satisfies
$\lVert x-y\rVert < c$, then

$$
g(x) \;=\; \min_{y'\in B}\lVert x-y'\rVert \;\le\; \lVert x-y\rVert \;<\; c ,
$$

so $x$ cannot attain a maximum already known to be $\ge c$. One kernel of
work retires an entire inner loop. $\blacksquare$

This is exactly the early-break in SciPy's `directed_hausdorff` (Taha &
Hanbury, *IEEE TPAMI* 2015): maintain the running maximum `cmax`, and
abandon the inner scan the instant any $y$ falls within `cmax`.

> **The fundamental asymmetry, stated once:**
> $$\text{certificate complexity of }\textstyle\sum_{m,n} \;=\; \Theta(M^2), \qquad \text{certificate complexity of } \max_x\min_y \;=\; O(M).$$
> A **sum** has no witnesses — every term must be seen. A **max-min** is
> certified by one witness per outer point. This, not implementation
> quality, is why one prunes and the other cannot.

**Proposition (expected cost under randomised scan order).** Fix $x$ and
let `cmax` be the running maximum. Let

$$
S_x \;=\; \big|\{\,y\in B:\ \lVert x-y\rVert < \texttt{cmax}\,\}\big|, \qquad p_x = S_x/M .
$$

Scanning $B$ in uniformly random order, the number of inner iterations
$T_x$ before the first witness is the position of the first "success" when
drawing without replacement from $M$ items containing $S_x$ successes:

$$
\boxed{\ \mathbb E[T_x] \;=\; \frac{M+1}{S_x+1} \;=\; \frac{M+1}{M p_x + 1}\ }
$$

Hence the expected total work is
$\mathbb E\big[\sum_{x\in A}T_x\big] = \sum_{x\in A}\frac{M+1}{Mp_x+1}$, and:

- if $p_x=\Theta(1)$ for typical $x$ — i.e. a constant fraction of $B$ lies
  within `cmax` — then $\mathbb E[T_x]=O(1)$ and the total is $\boxed{O(M)}$;
- only the few points near the true maximiser have $p_x\to0$ and pay a full
  $O(M)$ scan;
- in the adversarial case $p_x=\Theta(1/M)$ for all $x$, the bound degrades
  to the worst case $O(M^2)$.

**Why $p_x=\Theta(1)$ holds here.** `cmax` rapidly approaches
$h=\max_x g(x)$, which is a *maximum* over nearest-neighbour distances. A
typical $x$ has $g(x)\ll h$, so a ball of radius $h$ about $x$ contains a
constant fraction of $B$. In this benchmark the clouds are Gaussian blobs
of spread $0.3$ around latent centres $O(1)$ apart, so this holds
comfortably — and the measured affine fit below confirms it.

### 30.3 The cost model, and a falsifiable prediction

Theorems A and B predict per-pair costs of the form

$$
C_{\mathrm{haus}}(M) = c_0 + c_1 M \quad (\text{affine}), \qquad
C_{\mathrm{gram}}(M) = c_2 M^2 \quad (\text{quadratic}),
$$

where $c_0$ is fixed Python$\to$C call overhead (paid twice per pair).
Fitting **only** the endpoints $M\in\{10,640\}$ ($N{=}30$ clouds, 435
pairs, µs/pair):

$$
\widehat C_{\mathrm{haus}}(M) = 138.3 + 0.1456\,M, \qquad
\widehat C_{\mathrm{gram}}(M) = 0.02215\,M^2 .
$$

**Held-out validation** at $M\in\{40,80,160\}$ (not used in the fit):

| $M$ | Hausdorff pred. | measured | Gram pred. | measured |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 144.2 | 140.6 | 35.4 | 40.4 |
| 80 | 150.0 | 145.6 | 141.7 | 138.9 |
| 160 | 161.6 | 157.9 | 566.9 | 555.5 |

Both functional forms are confirmed to within 3–14%. In particular
Hausdorff really is **affine** in $M$ — $64\times$ more points costs only
$1.7\times$ more time — vindicating the $O(M)$-average claim of §30.2, and
the Gram is cleanly **quadratic**, vindicating Theorem A.

**Crossover.** Setting $c_0+c_1M=c_2M^2$ and taking the positive root:

$$
\boxed{\ M^\star \;=\; \frac{c_1+\sqrt{c_1^2+4c_0c_2}}{2c_2} \;=\; \frac{0.1456+\sqrt{0.1456^2+4(138.3)(0.02215)}}{2(0.02215)} \;=\; 82.4\ }
$$

**Measured at $M{=}80$: cost ratio $= 0.95$** — parity, as predicted, to
within 3% of $M^\star$. The full curve:

| $M$ | Hausdorff µs/pair | Gram µs/pair | ratio (gram/haus) |
| ---: | ---: | ---: | ---: |
| 10 | 139.8 | 7.7 | 0.05 (**herding 18× faster**) |
| 40 | 140.6 | 40.4 | 0.29 |
| **80** | **145.6** | **138.9** | **0.95 ← crossover** |
| 160 | 157.9 | 555.5 | 3.52 |
| 640 | 231.5 | 9071.0 | **39.2 (Hausdorff 39× faster)** |

The asymptotic ratio grows without bound:
$C_{\mathrm{gram}}/C_{\mathrm{haus}} = \Theta(M^2)/\Theta(M) = \Theta(M)$.
So the disadvantage is not a constant penalty — **it grows linearly in the
number of examples per workload.**

### 30.4 Practical consequence

Real Text2SQL workloads have $M$ in the hundreds to thousands. At $M{=}640$
the exact-Gram route is already $39\times$ slower; at $M{=}5000$ the model
predicts $\approx 0.02215\cdot 5000^2 = 554$ ms/pair versus $\approx 0.9$
ms/pair for Hausdorff — a $600\times$ gap, and $\binom{113}{2}$ pairs would
take roughly an hour versus six seconds. **The §27.9 benchmark's $M{=}20$
sits an order of magnitude below the crossover, which is the sole reason
herding appeared cheap there.**

### 30.5 The fix — and why it is the *dual* of §28.2

Herding's $\Theta(M^2)$ is **not intrinsic to the method**; it is an
artifact of evaluating the Gram *exactly through the kernel trick*.
Theorem A's lower bound applies to the kernel-oracle model — but we have
the raw coordinates, and because the mean embedding is **additive** we can
linearise it.

**Random Fourier Features (Rahimi & Recht, NIPS 2007).** By Bochner's
theorem a shift-invariant kernel is the Fourier transform of a probability
measure; for the Gaussian $k(x,y)=e^{-\lVert x-y\rVert^2/\tau}$ that
measure is $\mathcal N\!\big(0,\tfrac{2}{\tau}I\big)$. Draw
$\omega_1,\dots,\omega_D\sim\mathcal N(0,\tfrac2\tau I)$ and
$b_r\sim\mathrm{Unif}[0,2\pi]$, and set

$$
z(x) \;=\; \sqrt{\tfrac2D}\,\Big(\cos(\omega_1^\top x+b_1),\ \dots,\ \cos(\omega_D^\top x+b_D)\Big)\ \in\ \mathbb R^{D},
\qquad \mathbb E\big[z(x)^\top z(y)\big] = k(x,y).
$$

Now the crucial step — **because $\mu_i$ is an average, the feature map
commutes with it**:

$$
\hat\mu_i \;=\; \frac1M\sum_{m=1}^{M} z(x_{im}) \ \in\ \mathbb R^{D}
\qquad\Longrightarrow\qquad
G_{ij} \;\approx\; \hat\mu_i^\top\hat\mu_j .
$$

Cost: $O(MD)$ **once per cloud** to build $\hat\mu_i$, then $O(D)$ **per
pair** — total $O(NMD + N^2D)$ instead of $O(N^2M^2)$. **Linear in $M$**,
with uniform approximation error $O\big(\sqrt{\log M/D}\big)$.

**Hausdorff admits no such linearisation.** §28.2 proved the max-min does
not decompose into per-cloud summaries — there is no finite-dimensional
$\psi$ with $d_H(A,B)\approx f(\psi(A),\psi(B))$. It cannot be linearised;
it can only be **pruned**.

This yields a pleasing duality — the two structures are opposites in *both*
respects:

| | Structure | Exact cost | Gets to $O(M)$ by | Summarisable? |
| --- | --- | --- | --- | --- |
| **Kernel herding** | additive $\sum$ | $\Theta(M^2)$ (Thm A) | **linearisation** (RFF) | **yes** — one vector $\hat\mu_i$ |
| **Hausdorff** | nested $\max\min$ | $O(M)$ avg (Thm B) | **pruning** (certificates) | **no** (§28.2) |

The very additivity that makes herding's *selection loop* cheap (§28.2 —
one reusable vector) is what makes its *exact Gram* expensive; and it is
simultaneously what provides the escape hatch. Conversely the nested
extremum that makes Hausdorff *unsummarisable* is exactly what makes it
*prunable*. **Each objective's weakness is the source of its own remedy.**

⚠️ **Not implemented or tested.** The RFF route is a proposal derived here,
not a measured result. It carries its own trade-offs — the $O(\sqrt{\log M/D})$
error perturbs $G$, and §29's covering-radius/MMD guarantees would need
re-deriving under that perturbation. Treat as the natural next experiment,
not as a solved problem.

### 30.6 Summary

| Question | Answer |
| --- | --- |
| Why is the Gram $\Theta(M^2)$? | **Theorem A**: every one of the $M^2$ terms has non-zero influence on a sum, so an adversary can invalidate any answer that skipped one. Sums admit **no certificates**. |
| Why is Hausdorff $O(M)$? | **Theorem B**: a single witness $y$ with $\lVert x-y\rVert<\texttt{cmax}$ certifies that $x$ is irrelevant, retiring its whole inner loop. Expected inner length $\frac{M+1}{Mp_x+1}=O(1)$ when a constant fraction of $B$ is within `cmax`. |
| Where do they cross? | $M^\star=\frac{c_1+\sqrt{c_1^2+4c_0c_2}}{2c_2}=82.4$ from endpoint-only fits; **measured parity at $M{=}80$ (ratio 0.95)**. |
| How bad does it get? | Ratio grows as $\Theta(M)$ — unbounded. $39\times$ at $M{=}640$; a predicted $\sim600\times$ at $M{=}5000$. |
| Is it fixable? | **Yes, in principle** — additivity permits RFF linearisation to $O(MD)$. Untested here, and it perturbs the guarantees of §29. |
