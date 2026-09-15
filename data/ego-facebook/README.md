# ego-Facebook -- real-world graphs used for validation

Three ego-networks from the same SNAP archive are used, to move validation
beyond a single real graph (N=1): **348** (the original), **686**, and
**3437**.

- **Source**: SNAP, Stanford Network Analysis Project --
  https://snap.stanford.edu/data/ego-Facebook.html
- **Direct file**: https://snap.stanford.edu/data/facebook.tar.gz
  (this repo keeps only `348.edges`/`686.edges`/`3437.edges`, extracted
  from that archive; the archive also ships `.circles`/`.feat`/
  `.featnames`/`.egofeat` files for each ego, none of which are used here
  -- see below).
- **Citation**: J. McAuley and J. Leskovec, "Learning to Discover Social
  Circles in Ego Networks," NIPS, 2012.
- **What these files are**: the induced friendship subgraph among each
  ego-user's friends (the ego node itself is excluded, per SNAP's own
  convention for this dataset).

| ego | nodes | undirected edges | avg degree | connected? |
|---|---|---|---|---|
| 348  | 224 | 3,192 | ~28.5 | single component |
| 686  | 168 | 1,656 | ~19.7 | single component |
| 3437 | 534 | 4,813 | ~18.0 | single component |

All three are real, publicly available, single-connected-component, and
well under the ~500-1000-node budget these Monte-Carlo-heavy algorithms
(CELF-greedy, IMM, Saturate-Greedy/robust_kempe, repeated_greedy,
MF-BWI-Fair) can stay tractable on.
- **Why these three egos and not the other seven**: of the 10 ego-networks
  shipped in the archive, sizes range from 52 to 1,034 nodes; only five
  (107, 348, 686, 1912, 3437) are a SINGLE connected component (the rest
  have small disconnected islands that `im_lab.graphs.load_edge_list_graph`
  would otherwise have to drop, silently shrinking the graph below its
  stated node count). Within that fully-connected subset, 348/686/3437
  were picked for size diversity (168 to 534 nodes) while excluding the
  two largest/densest (107: 1,034 nodes, avg degree ~51.7; 1912: 747
  nodes, avg degree ~80.4), which would make the full study-matched
  compute budget (`experiments/MULTIGRAPH_VALIDATION.md`) intractable in
  this project's execution environment.
- **What is deliberately NOT used from the archive**: the `.circles` /
  `.feat` / `.featnames` files encode real (anonymized) Facebook "friend
  list" circles and profile features for this ego's friends. This project
  does not use them as fairness-group labels, specifically because they are
  the closest thing in this dataset to real demographic/affiliation
  information about real people, and repurposing them as "fairness groups"
  in an algorithm-comparison experiment would misrepresent structural
  proxy groups as real demographic ones. Group labels for the fairness
  dimension are instead produced structurally, via community detection
  (`im_lab.graphs.assign_communities_as_groups`) -- see
  `experiments/MULTIGRAPH_VALIDATION.md` for the explicit "this is a proxy,
  not real demographic data" note repeated at the point of use.
