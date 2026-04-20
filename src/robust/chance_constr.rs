// Chance-constrained MIP formulation.
//
// Ref: Blackmore, Ono & Williams (2011), Calafiore & Campi (2006).
//
// Given N scenarios ω₁,...,ωₙ (wind, atmosphere, Cd):
//
//   min  fuel(x) + λ·Σ z_i
//   s.t.
//     Σ z_i ≤ ⌊ε·N⌋               (at most ε fraction discarded)
//     g(x, ωᵢ) ≤ M·z_i             (big-M: active when z_i=0)
//     z_i ∈ {0,1}
//
// Practical decomposition (avoids giant monolithic MIP):
//   1. Solve nominal SCvx → reference trajectory x̄
//   2. For each scenario: linearized sensitivity ∂g/∂x|_{x̄, ωᵢ}
//   3. MIP master: select z_i, adjust x from reference
//   4. Re-solve SCvx with tightened constraints
//   5. Iterate until convergence
