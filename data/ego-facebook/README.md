# ego-Facebook (node 348) -- real-world graph used for validation

- **Source**: SNAP, Stanford Network Analysis Project --
  https://snap.stanford.edu/data/ego-Facebook.html
- **Direct file**: https://snap.stanford.edu/data/facebook.tar.gz
  (this repo keeps only `348.edges`, extracted from that archive; the
  archive also ships `.circles`/`.feat`/`.featnames`/`.egofeat` files for
  each ego, none of which are used here -- see below).
- **Citation**: J. McAuley and J. Leskovec, "Learning to Discover Social
  Circles in Ego Networks," NIPS, 2012.
- **What this file is**: the induced friendship subgraph among the Facebook
  ego-user "348"'s friends (the ego node itself is excluded, per SNAP's own
  convention for this dataset) -- 224 nodes, 3,192 undirected edges (listed
  once per direction in the raw file, 6,384 lines total), a single connected
  component, average degree ~28.5. Real, publicly available, and small
  enough (well under the ~500-node budget) for this project's Monte-Carlo-
  heavy algorithms (CELF-greedy, IMM, Saturate-Greedy/robust_kempe,
  repeated_greedy, MF-BWI-Fair) to stay tractable.
- **Why this ego (348) and not another**: of the 10 ego-networks shipped in
  the archive, sizes range from 52 to 1,034 nodes. 348 was chosen because it
  is (a) a moderate few-hundred-node size close to this project's existing
  120-node synthetic graph, (b) a SINGLE connected component (several of the
  others have small disconnected islands that `im_lab.graphs.load_edge_list_graph`
  would otherwise have to drop), and (c) dense enough (avg degree ~28.5) that
  a weighted-cascade probability assignment (`p = 1/in-degree`, typically
  ~0.035 here) still produces a real cascade rather than dying out
  immediately.
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
