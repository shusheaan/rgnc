// Trajectory library optimization via MIP set cover.
//
// Ref: facility location / set cover formulation.
//
// Given M candidate trajectories, N scenarios, performance matrix d[i][j]:
//
//   select_j ∈ {0,1}            (trajectory j in library?)
//   assign_{ij} ∈ {0,1}         (scenario i served by trajectory j?)
//
//   min  t                       (minimize worst-case error)
//   s.t.
//     Σ_j select_j ≤ K          (at most K trajectories)
//     Σ_j assign_{ij} ≥ 1  ∀i   (every scenario covered)
//     assign_{ij} ≤ select_j     (can only assign to selected)
//     Σ_j d[i][j]·assign_{ij} ≤ t  ∀i  (worst case bounded)
