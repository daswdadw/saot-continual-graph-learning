# Chapter 3. Optimal Transport and Structure-Aware Alignment

## 3.1 Why a transport plan is useful

SAOT needs a common object for comparing two graph views in spaces with very
different semantics. In the graph space, node attributes and edges are directly
observable. In the representation space, the encoder produces vectors whose
pairwise geometry changes during learning. A transport plan provides that common
object. Its entry \(\pi_{ij}\) is the amount of probability mass assigned from
node \(i\) in the first view to node \(j\) in the second view. With empirical
masses \(\mu\) and \(\nu\), every feasible plan satisfies

\[
\Pi(\mu,\nu)=\{\pi\ge 0\mid \pi\mathbf{1}=\mu,
\pi^\top\mathbf{1}=\nu\}.
\]

The row and column constraints prevent the solver from matching every node to a
single easy target. SAOT uses uniform empirical masses in its two augmented
views, so each node contributes the same total mass.

## 3.2 Wasserstein optimal transport

For entities that already share a metric space, discrete Wasserstein OT solves

\[
D(X^1,X^2)=\min_{\pi\in\Pi(\mu,\nu)}
\langle K(X^1,X^2),\pi\rangle,
\]

where \(K_{ij}=c_X(x_i^1,x_j^2)\) is a cross-view feature cost, such as cosine
distance. A low-cost pair receives more mass, subject to both marginals. The
representation-space plan in SAOT is exactly this construction:

\[
\pi_Z^*=\arg\min_{\pi_Z\in\Pi(\mu,\nu)}
\langle R(Z^1,Z^2),\pi_Z\rangle,
\]

where \(R_{ij}=c_Z(z_i^1,z_j^2)\). In our implementation, entropic regularization
makes the objective smooth and enables a stable log-domain Sinkhorn solver.

## 3.3 Gromov-Wasserstein transport

Ordinary Wasserstein distance cannot compare graph topology by itself because
nodes in different graphs do not share coordinates. Gromov-Wasserstein (GW)
instead asks whether *within-graph relations* are preserved under a coupling.
For adjacency-derived relation matrices \(A^1\) and \(A^2\), SAOT uses the
structural discrepancy

\[
c_A(A^1_{ik},A^2_{jl})=|A^1_{ik}-A^2_{jl}|.
\]

The GW objective is

\[
D_{GW}(A^1,A^2)=\min_{\pi\in\Pi(\mu,\nu)}
\sum_{i,j,k,l}c_A(A^1_{ik},A^2_{jl})\pi_{ij}\pi_{kl}.
\]

The quadratic term is the key: assigning \(i\) to \(j\) is evaluated together
with assigning \(k\) to \(l\). Thus two nodes can match well even when their raw
features differ, provided their neighborhoods play analogous roles.

## 3.4 Fused Gromov-Wasserstein in graph space

Fused Gromov-Wasserstein (FGW) combines direct feature evidence with structural
evidence. SAOT defines the graph-space plan as

\[
\pi_G^*=\arg\min_{\pi_G\in\Pi(\mu,\nu)}
\left[
\sigma\sum_{ij}K_{ij}\pi^G_{ij}
+(1-\sigma)\sum_{ijkl}
|A^1_{ik}-A^2_{jl}|\pi^G_{ij}\pi^G_{kl}
\right].
\]

The parameter \(\sigma\in[0,1]\) has a direct interpretation:

- \(\sigma=0\): matching is determined only by structural roles;
- \(0<\sigma<1\): attributes and topology jointly determine the plan;
- \(\sigma=1\): FGW reduces to feature-only Wasserstein OT.

This convention is important because some FGW libraries define their mixing
parameter in the opposite direction. Our solver follows the paper: a larger
\(\sigma\) means more feature weight.

## 3.5 Numerical solver

The graph-space objective is quadratic in \(\pi_G\). We solve an
entropy-regularized fixed-point approximation. Starting from the independent
coupling \(\mu\nu^\top\), each outer iteration contracts the four-index
structural cost tensor under the current plan. The resulting local cost is

\[
C^{(t)}=\sigma K+2(1-\sigma)[L(A^1,A^2)\otimes\pi^{(t)}].
\]

A log-domain Sinkhorn step then returns a feasible coupling for \(C^{(t)}\). We
optionally relax the update to prevent oscillation. Iteration stops when the
L1 plan change is below the tolerance. The implementation reports marginal
error, plan change, individual feature and structure terms, and the full
objective trace. The paper's absolute structural loss requires
\(O(n^2m^2)\) memory in the direct contraction; a squared-loss alternative is
included for larger sampled graphs because it can be contracted without
materializing the four-index tensor.

## 3.6 Toy-graph verification

The verification experiment creates two eight-node graphs with a known node
permutation, two dense communities, a bridge, mild feature noise, and one
view-specific edge perturbation. It evaluates \(\sigma\in\{0,0.25,0.5,0.75,1\}\),
saves every coupling as CSV and NPY, and outlines the known correspondence in
each heatmap. The experiment also computes \(\pi_Z\) from synthetic embeddings.
Three checks make the result auditable:

1. each plan's row and column sums match the uniform marginals;
2. structurally or semantically corresponding nodes receive the largest mass;
3. changing \(\sigma\) changes the plan and the feature/structure contributions.

The experiment is intentionally independent of the complete SAOT training
script. This isolates the OT component and gives the core-loss owner a small,
tested interface for computing \(\pi_G\) and \(\pi_Z\).
