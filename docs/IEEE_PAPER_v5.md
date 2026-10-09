# EcoLinkAI: A Multi-Channel Graph Neural Network for Industrial Waste Exchange Partner Recommendation

**Anavadya Nair** — Department of Computer Science and Engineering, Mar Baselios College of Engineering and Technology, Thiruvananthapuram, India — anavadyanair.b23cs1113@mbcet.ac.in

**Amith Shaji George** — Department of Computer Science and Engineering, Mar Baselios College of Engineering and Technology, Thiruvananthapuram, India — amithshajigeorge.b23cs1111@mbcet.ac.in

**Anirudh M Nair** — Department of Computer Science and Engineering, Mar Baselios College of Engineering and Technology, Thiruvananthapuram, India — anirudhmnair.b23cs1114@mbcet.ac.in

**Athul S.** — Department of Computer Science and Engineering, Mar Baselios College of Engineering and Technology, Thiruvananthapuram, India — athuls.b23cs1121@mbcet.ac.in

**Dr. Jesna Mohan** — Department of Computer Science and Engineering, Mar Baselios College of Engineering and Technology, Thiruvananthapuram, India — jesna.mohan@mbcet.ac.in

---

## Abstract

Industrial symbiosis enables waste and by-products from one industrial plant to serve as resources for another, but identifying suitable exchange partners requires jointly considering material compatibility, quantity, geographical distance, transportation cost, environmental benefit, and prior trading relationships. This paper presents EcoLinkAI, an industrial symbiosis platform that formulates waste-exchange partner recommendation as supervised link prediction over an industrial exchange graph using a Multi-Channel Graph Neural Network (MC-GNN). Each of the three channels reads a different relationship network — trading relationships (GAT), processing capability (GraphSAGE), and industrial similarity (GCN) — and the channels are fused by learned per-node attention with channel dropout. We show that the conventional inner-product Hilbert-Schmidt Independence Criterion (HSIC) regularizer, at its customary weight of $10^{-10}$, does not influence training, and that a normalized HSIC reduces the similarity between channel representations from 0.487 to 0.005. Because embedding each facility independently cannot represent what a supplier and a receiver share, pairs are scored by a pair-aware head that pools the embeddings of facilities linked to both. On six years (2019–2024) of real off-site recycling and energy-recovery transfers from the US EPA Toxics Release Inventory, trained on 2020–2022, selected on 2023, and tested once on 2024 (23,448 recommendation queries), the MC-GNN achieves an AUC of 0.9891, NDCG@5 of 0.9754, Recall@5 of 0.9810, and top-1 accuracy (Hit@1) of 96.70%, compared with 88.58% for rule-based matching. For facilities with no transfer history, Hit@1 rises from 48.8% to 76.9%. Against XGBoost, the MC-GNN is significantly better in AUC under a paired bootstrap, while XGBoost retains a 0.23-point advantage in Hit@1. Real-trade statistics measured from the same data are used to calibrate the platform's market data, on which the deployed model reaches an AUC of 0.862 against 0.792 for rule-based matching.

**Index Terms** — Industrial Symbiosis, Circular Economy, Graph Neural Networks, Multi-Channel Graph Neural Networks, Waste Exchange, Link Prediction, Partner Recommendation, Hilbert-Schmidt Independence Criterion

---

## I. Introduction

Industrial facilities generate by-products and waste streams that may retain value as inputs for other industrial processes. Industrial symbiosis seeks to establish such exchanges, reducing waste disposal and the need for virgin resources. However, identifying suitable exchange partners is challenging because a feasible transaction depends on multiple factors, including material compatibility, quantity requirements, geographical distance, transportation cost, environmental benefit, and prior trading relationships.

Existing matching approaches commonly rely on manual selection or weighted scoring functions that evaluate candidate pairs independently. Such approaches capture direct pairwise characteristics but do not represent the relationships among plants and their historical exchanges. Industrial exchange platforms naturally form a network in which plants act as nodes and supplier-to-receiver transfers form directed edges, which motivates a graph-based formulation of partner recommendation as supervised link prediction.

EcoLinkAI addresses this problem with a Multi-Channel Graph Neural Network (MC-GNN) [1] whose GCN, GAT, and GraphSAGE channels [2]–[4] are fused through learned per-node attention and regularized with the Hilbert-Schmidt Independence Criterion (HSIC) [14]. Evaluating this design on real transfer records exposed three weaknesses, each addressed in this work. First, when all channels read the same graph they learn largely redundant representations. Second, the HSIC term at the weight used in prior work is numerically too small to affect training. Third, a scorer that concatenates two independently computed node embeddings cannot represent what a supplier and a receiver share [16].

Real exchange records are scarce in industrial symbiosis research. We therefore evaluate on six years of facility-to-facility recovery transfers reported to the US EPA Toxics Release Inventory (TRI) [19], which record which facility sent which chemical, in what quantity, to which recycling or energy-recovery facility. The same records show how real trade behaves — most notably that facilities return to existing partners — and these measurements are used to calibrate the market data on which the platform's own model is trained.

The main contributions of this work are:

- A multi-view MC-GNN in which each channel reads a distinct relationship network — trading relationships, processing capability, and industrial similarity — with channel dropout to keep all channels in use.
- A normalized HSIC disparity term, with evidence that the conventional formulation does not influence training and that the normalized one produces nearly independent channel representations.
- A pair-aware scoring head, adapted from neural common-neighbor pooling [15] to supplier–receiver networks, in which the informative shared neighborhood lies at higher order.
- An evaluation on six years of real EPA TRI recovery transfers under a strict temporal protocol, comparing graph and non-graph learners and rule-based matching, with paired bootstrap significance testing and a separate analysis of facilities with no transfer history.
- An operational platform whose market data is calibrated to real trade statistics, with a deployed recommender, an exchange feedback loop, and a guarded model-promotion process.

---

## II. Related Work

Graph neural networks (GNNs) have been widely applied to recommendation, matching, and relational learning. Graph Convolutional Networks (GCNs) learn node representations through neighborhood aggregation, Graph Attention Networks (GATs) assign different importance to neighboring nodes, and GraphSAGE provides an inductive sample-and-aggregate mechanism [2]–[4].

Recent studies have investigated GNN-based matching and recommendation in different application domains. Afrin et al. explored graph-based surplus-demand matching, while Lan et al. applied GCNs to recommendation under sparse interactions. Graph contrastive learning and diffusion-based graph models have also been investigated for recommendation [6], [8], [10], [11]. The MC-GNN architecture combines multiple graph channels with attention and HSIC-based representation regularization [1], [14].

For link prediction specifically, GNNs that embed each node independently cannot count the common neighbors of two nodes, because each node's representation is computed without knowledge of the other [16]. Neural Common Neighbor (NCN) [15] addresses this while retaining a single message-passing run, by pooling the embeddings of node sets that depend on the target pair. Centered kernel alignment, a normalized form of HSIC, measures the similarity of two learned representations independently of their scale [17].

GNNs have also been applied to industrial and supply-chain problems, including supply-chain risk prediction, optimization, circular supply chains, and industrial IoT analysis [5], [7], [13]. Within industrial symbiosis, recommender systems and text-based approaches have been investigated for facilitating waste-to-resource exchanges. These studies do not combine multi-channel graph representation learning with pair-aware link prediction, nor evaluate on real facility-to-facility transfer records.

---

## III. Proposed Methodology

### A. Problem Formulation

The industrial exchange ecosystem is represented as a graph

$$G = (V, E), \tag{1}$$

where $V$ denotes facilities and $E$ denotes historical transfers. A query is a supplier $s$ and a waste stream it must place; the task is to rank candidate receivers $b$ so that those actually used appear first. Each candidate pair carries a label $y \in \{0, 1\}$. All graph structure and features used for a query are computed only from transfers before the period being predicted.

### B. Node and Edge Features

Each facility is represented by a 17-dimensional vector: geographic location, trust, number of distinct waste streams sent, number received, transfer activity, rating, and a ten-dimensional industry-sector indicator. Each supplier–receiver pair is represented by an eight-dimensional vector

$$\mathbf{e}_{s,b} = [d, m, q, r, c, s, p, h], \tag{2}$$

with normalized distance $d$, material compatibility $m$, quantity compatibility $q$, quality compatibility $r$, transport cost $c$, carbon saving $s$, prior successful transfers $p$, and a prior-interaction indicator $h$, all scaled to $[0, 1]$. The same encoder is used for training and serving.

### C. Relationship Networks

Each MC-GNN channel reads its own network, built from history only:

- **Relationship network:** facilities connected by a past transfer, in both directions.
- **Capability network:** each facility's outgoing waste profile linked to the facilities whose incoming profile is most similar, and vice versa.
- **Peer network:** facilities similar in detailed industry classification (NAICS at 3 to 6 digits) and in the waste streams they send or receive.

The capability and peer networks are $k$-nearest-neighbor graphs ($k = 10$) over IDF-weighted profiles, so that sharing a rare chemical counts more than sharing a common one. For the capability network,

$$\mathrm{sim}(a, b) = \cos(\mathbf{o}_a, \mathbf{i}_b), \tag{3}$$

where $\mathbf{o}_a$ and $\mathbf{i}_b$ are the outgoing profile of $a$ and the incoming profile of $b$. Two pair-level quantities are derived from these networks and appended to the edge vector: the capability score $\cos(\mathbf{o}_s, \mathbf{i}_b)$ and the share of the supplier's peers that previously sent to the candidate receiver.

### D. Multi-Channel Encoder

The GAT channel reads the relationship network, where partners differ in importance; GraphSAGE reads the capability network, whose large neighborhoods suit mean aggregation; GCN reads the peer network, where similar facilities count roughly equally. Each channel uses two graph layers and a linear skip connection from the input features:

$$\mathbf{Z}_c = f_c^{(2)}\left(\mathrm{ReLU}\left(f_c^{(1)}(\mathbf{X}, \mathbf{A}_c)\right), \mathbf{A}_c\right) + \mathbf{W}_c^{skip}\mathbf{X}, \tag{4}$$

where $\mathbf{A}_c$ is the network read by channel $c$. The hidden dimension is 128 and the output dimension 64.

### E. Attention Fusion with Channel Dropout

For channel $i$ and node $v$,

$$w_i^v = \mathbf{q}^T \tanh\left(\mathbf{W}_i \mathbf{z}_i^v + \mathbf{b}_i\right), \qquad \alpha_i^v = \mathrm{softmax}_i(w_i^v), \qquad \mathbf{z}^v = \sum_i \alpha_i^v \mathbf{z}_i^v. \tag{5}$$

Without intervention, attention concentrated on a single channel for nearly every node (about 94% of the weight on one network). During training, for each node, with probability $p_{cd} = 0.3$ one channel $k$ is excluded from the softmax:

$$\alpha_i^v = \frac{\exp(w_i^v)\,\mathbb{1}[i \neq k]}{\sum_j \exp(w_j^v)\,\mathbb{1}[j \neq k]}, \tag{6}$$

so that every channel must carry useful information. Channel dropout is disabled at inference.

### F. Normalized HSIC Regularization

HSIC with inner-product kernels [1], [14] is

$$\mathrm{HSIC}(\mathbf{Z}_i, \mathbf{Z}_j) = \frac{1}{(N-1)^2}\mathrm{tr}(\mathbf{K}_i \mathbf{H} \mathbf{K}_j \mathbf{H}), \qquad \mathbf{K}_i = \mathbf{Z}_i \mathbf{Z}_i^T, \quad \mathbf{H} = \mathbf{I} - \tfrac{1}{N}\mathbf{1}\mathbf{1}^T. \tag{7}$$

This quantity grows with the squared magnitude of the embeddings, so it can be reduced by shrinking the embeddings without making the channels more different, and its weight must be very small (here $10^{-10}$) to remain below the task loss. On trained models, the weighted term was approximately $10^{-11}$ against a cross-entropy near 0.07: the regularizer did not influence training. We therefore use the normalized form

$$\mathrm{nHSIC}(\mathbf{Z}_i, \mathbf{Z}_j) = \frac{\mathrm{HSIC}(\mathbf{Z}_i, \mathbf{Z}_j)}{\sqrt{\mathrm{HSIC}(\mathbf{Z}_i, \mathbf{Z}_i)\,\mathrm{HSIC}(\mathbf{Z}_j, \mathbf{Z}_j)}}, \tag{8}$$

which equals linear centered kernel alignment [17], lies in $[0, 1]$, and is invariant to rescaling. The disparity term averages nHSIC over the three channel pairs, computed on 2,048 randomly sampled nodes per step:

$$\mathcal{L}_{HSIC} = \frac{1}{3}\sum_{i<j} \mathrm{nHSIC}(\mathbf{Z}_i, \mathbf{Z}_j). \tag{9}$$

### G. Pair-Aware Scoring

In a supplier–receiver network, suppliers trade with receivers rather than with each other, so the common neighbors of a supplier and a receiver are rarely present. On the 2024 test pairs, direct common neighbors exist for 8.3% of pairs and separate true partners from alternatives at an AUC of only 0.613. The informative neighborhood lies at higher order, which the general form of NCN accommodates [15]. For each pair we construct two sets:

- **Co-partners** $C(s,b) = N(b) \cap N^2(s)$: facilities that share a trading partner with the supplier and already trade with the receiver. Present for 36.2% of test pairs; their count alone separates true partners at AUC 0.890.
- **Peers** $P(s,b) = N_{peer}(s) \cap N(b)$: the supplier's peers that already trade with the receiver (AUC 0.735).

Each set is truncated to its 16 members most strongly tied to the supplier, with ties through a shared partner weighted by $1/\log(1 + \deg)$ of that partner, so that sharing a specialized receiver counts more than sharing one that accepts from thousands of facilities. The score is

$$p(s \to b) = \sigma\left(\mathrm{MLP}\left[\mathbf{z}^s \,\|\, \mathbf{z}^b \,\|\, \mathbf{z}^s \odot \mathbf{z}^b \,\|\, \overline{\mathbf{z}}_{C(s,b)} \,\|\, \overline{\mathbf{z}}_{P(s,b)} \,\|\, \mathbf{e}_{s,b} \,\|\, \mathbf{n}_{s,b}\right]\right), \tag{10}$$

where $\overline{\mathbf{z}}_S$ is the mean embedding of set $S$ and $\mathbf{n}_{s,b}$ the logarithms of the two set sizes. The head uses layers of 64 and 32 units with ReLU activations and dropout 0.2.

### H. Training Objective

Recommendations are judged per query, so a listwise ranking term is added to binary cross-entropy. For query $g$ with candidates $C_g$ and true partners $P_g$, and pre-sigmoid scores $\ell$,

$$\mathcal{L}_{rank} = -\frac{1}{\sum_g |P_g|}\sum_g \sum_{b \in P_g} \log \frac{\exp(\ell_{s,b})}{\sum_{b' \in C_g}\exp(\ell_{s,b'})}. \tag{11}$$

The overall objective is

$$\mathcal{L} = \mathcal{L}_{BCE} + \lambda \mathcal{L}_{rank} + \gamma \mathcal{L}_{HSIC}, \qquad \lambda = 1, \ \gamma = 0.2. \tag{12}$$

---

## IV. System Implementation

### A. Platform Architecture

EcoLinkAI is implemented as a web-based industrial symbiosis platform integrating data management, exchange services, and AI-based partner recommendation (Fig. 1). The client application provides interfaces for company and plant management, waste listings, requirements, exchanges, reviews, analytics, and partner recommendations. The backend uses FastAPI, SQLAlchemy, and PostgreSQL, and the AI components use PyTorch and PyTorch Geometric. The graph is constructed from the operational database, so recommendations reflect current plant, material, requirement, and exchange information.

*Fig. 1. Overall system architecture of the EcoLinkAI platform.*

### B. Recommendation Pipeline

A recommendation request generates candidate receivers from compatible open requirements, excluding the supplier's own company. For each candidate pair the service computes the same edge features used in training. To keep latency low, the constructed graph and the encoded node embeddings are cached and refreshed periodically, so each candidate requires only the scoring step (on the order of milliseconds) rather than re-encoding the graph. Candidates are returned ranked by probability, calibrated with Platt scaling on validation data, with explanations based on material compatibility, proximity, trust, prior dealings, and carbon benefit. A weighted rule-based scorer serves as a fallback when the model is unavailable.

### C. Exchange Feedback Loop and Model Promotion

An accepted recommendation creates an exchange request; completed exchanges generate transaction and review records. Reviews update company trust, and accepted exchanges become relationships for subsequent graph construction and retraining. A retrained model replaces the serving model only if it improves held-out AUC by more than 0.02 on the same temporal split, and the replacement is performed atomically so that the service never loads a partially written model.

### D. Market Data Calibrated to Real Trade

The platform's dataset combines real facility records from the KINFRA industrial park with generated market counterparties. In the first version of this market, each buyer approached each seller once, so a pair's own trading history could never influence a decision. The prior-interaction feature separated accepted from rejected requests at an AUC of 0.508 there, against 0.924 for true partners in the EPA TRI data. We therefore measured, from the TRI training years, how real recovery trade decides between partners, and rebuilt the market generator to follow those measurements (Table VII). Buyers now re-order in successive rounds, return to existing partners with a probability chosen to reproduce the measured retention, and accept offers according to the measured weights. The real KINFRA companies and their listings are unchanged; the generated trades are synthetic and are used for training and demonstration, not as evidence of real-world accuracy.

---

## V. Experimental Setup

### A. Dataset

The evaluation uses US EPA TRI off-site transfer records for 2019–2024 [19], retaining only transfers for recycling and energy recovery. Quantities are converted to kilograms. Sending facilities carry reported coordinates; receiving facilities are located through the EPA Facility Registry Service (4,631 facilities), TRI self-reported coordinates (748), or ZIP-code centroids (216), with 179 not located.

**TABLE I — EPA TRI DATASET**

| Year | Role | Transfers | Both ends located | Facilities | Queries | True pairs | Sampled non-pairs |
|---|---|---|---|---|---|---|---|
| 2019 | History | 42,848 | 97.7% | – | – | – | – |
| 2020 | Training | 41,335 | 98.1% | 12,855 | 23,986 | 38,737 | 1,741,272 |
| 2021 | Training | 41,448 | 98.0% | 13,863 | 23,943 | 39,038 | 1,756,244 |
| 2022 | Training | 41,459 | 98.0% | 14,708 | 24,228 | 39,145 | 1,760,474 |
| 2023 | Validation | 41,308 | 97.9% | 15,556 | 23,900 | 39,267 | 353,403 |
| 2024 | Test | 40,709 | 98.3% | 16,263 | 23,448 | 38,845 | 349,605 |

Each query is a (supplier, chemical) pair in the target year. Candidates are receivers that in earlier years accepted the same chemical or used the same recovery method. As the records list only transfers that occurred, nine non-partners per true partner are sampled from the candidate pool for validation and testing, preferring candidates within the 90th percentile of historical transfer distance. For training, a pool of 45 non-partners per true partner is built and nine are drawn afresh each epoch. True partners with no history before the target year (2.9% in 2024) cannot appear in a candidate pool and are excluded. TRI contains no quality, trust, or rating information; these inputs take the neutral values the platform assigns to a company without records. All features, networks, and pair sets for a target year are built from earlier years only. Of the 23,448 test queries, 766 come from facilities with no transfer history.

### B. Protocol and Model Selection

Models learn from 2020–2022 (each year using all earlier years from 2019 as history), are selected on 2023, and are tested once on 2024. The MC-GNN is trained with Adam (learning rate 0.01, weight decay $10^{-4}$), up to 1,000 epochs, with early stopping on validation AUC (patience 40). Hyperparameters were chosen on 2023 data only:

**TABLE II — MODEL SELECTION ON 2023 VALIDATION DATA (ONE SEED)**

| Setting searched | Values (validation result) | Chosen |
|---|---|---|
| HSIC weight $\gamma$ (hidden 64) | 0 (AUC 0.9888), 0.05 (0.9891), 0.2 (0.9895), 1.0 (0.9894) | 0.2 |
| Learning rate | 0.01 (AUC 0.9895), 0.003 (0.9893) | 0.01 |
| Ranking weight $\lambda$, size | 1.0, 64/32 (Hit@1 96.78%); 3.0, 64/32 (96.66%); 1.0, 128/64 (96.80%, AUC 0.9900); 3.0, 128/64 (96.70%) | 1.0, 128/64 |

The chosen configuration is trained with five random seeds, and the reported prediction is the mean of the five predicted probabilities.

### C. Compared Methods

- **XGBoost** [18]: trained on the 17 supplier features, 17 receiver features, and 8 pair features, with up to 3,000 trees, learning rate 0.05, maximum depth 6, row and column subsampling of 0.8, and early stopping on 2023 validation AUC (patience 40); five seeds, averaged as for the MC-GNN.
- **Single-channel GNNs** (GCN, GraphSAGE, GAT): each one MC-GNN channel on the relationship network, with the same pair-aware scoring head, inputs, and training procedure; three seeds, averaged.
- **MLP**: the same encoder depth and scoring head without message passing.
- **Rule-based baseline**: the platform's fixed weighted sum of the eight pair features.
- **Non-learning rankers**: past partner first, busiest receiver first, nearest receiver first, and random ranking.

XGBoost, the MLP, and the rule-based baseline use tabular features only; the graph models also read the relationship networks and two network-derived pair features.

### D. Metrics and Significance Testing

NDCG@5 and Recall@5 are computed within each query's candidate list and averaged over queries with at least one true partner. Hit@1 is the fraction of queries whose top-ranked candidate is a true partner. AUC measures ranking discrimination across all test pairs. Results are also reported for the 766 queries from facilities without history, the situation of a newly registered platform company. Differences are tested with a paired bootstrap over test queries (2,000 resamples), keeping each query's candidates together; a difference is significant when its 95% percentile interval excludes zero.

---

## VI. Results and Discussion

### A. Channel Diversity and HSIC

**TABLE III — SIMILARITY BETWEEN CHANNEL EMBEDDINGS (NORMALIZED HSIC; 0 = INDEPENDENT, 1 = IDENTICAL), 2024 TEST GRAPH, HIDDEN 64 / OUTPUT 32**

| Model | GCN–GAT | GCN–GraphSAGE | GAT–GraphSAGE | Mean |
|---|---|---|---|---|
| Multi-view MC-GNN, conventional HSIC ($\gamma = 10^{-10}$) | 0.451 | 0.613 | 0.398 | 0.487 |
| Multi-view MC-GNN, normalized HSIC ($\gamma = 0.2$) | 0.001 | 0.002 | 0.012 | 0.005 |

With the conventional term, the weighted HSIC contribution was about $10^{-11}$ against a cross-entropy of about 0.07, so the channel diversity that existed came from the separate networks and channel dropout rather than from the regularizer. With the normalized term, the channels learn nearly independent representations, and validation AUC rises from 0.9888 without HSIC to 0.9895 (Table II).

### B. Model Comparison

**TABLE IV — EPA TRI 2024 TEST (23,448 QUERIES)**

| Model | AUC | NDCG@5 | Recall@5 | Hit@1 | Hit@1 (new facilities) |
|---|---|---|---|---|---|
| **MC-GNN** | **0.9891** | **0.9754** | **0.9810** | **96.70%** | **76.9%** |
| XGBoost | 0.9887 | 0.9746 | 0.9790 | 96.93% | 75.6% |
| GCN | 0.9895 | 0.9754 | 0.9811 | 96.66% | 77.6% |
| GraphSAGE | 0.9891 | 0.9750 | 0.9808 | 96.58% | 76.1% |
| GAT | 0.9885 | 0.9742 | 0.9801 | 96.51% | 74.8% |
| MLP (no graph) | 0.9848 | 0.9660 | 0.9750 | 95.56% | 67.3% |
| Rule-based baseline | 0.9508 | 0.9129 | 0.9437 | 88.58% | 48.8% |
| Past partner first | 0.9487 | 0.9257 | 0.9295 | 92.48% | 44.1% |
| Busiest receiver first | 0.8272 | 0.6558 | 0.8150 | 48.19% | 43.1% |
| Nearest receiver first | 0.7343 | 0.5974 | 0.6639 | 52.73% | 44.1% |
| Random | 0.4999 | 0.2460 | 0.3954 | 10.07% | 8.5% |

*MC-GNN and XGBoost are averages of five runs; GCN, GraphSAGE, and GAT, each with the same pair-aware scoring head, are averages of three runs. The MLP is a mean of three runs trained with binary cross-entropy on the eight standard pair features.*

Every learned model substantially outperforms rule-based matching: by seven to eight percentage points in Hit@1 overall, and for facilities without history by 18.5 points for the MLP and 26–29 points for XGBoost and the graph models. Under identical training without the pair-aware head, every graph model — the base MC-GNN (96.22%), GCN (96.29%), GraphSAGE (96.08%), and GAT (95.93%) — outperformed the MLP (95.56%), which shares their features but has no message passing, confirming that the relationship networks carry predictive information on real data.

**TABLE V — PAIRED BOOTSTRAP: MC-GNN MINUS XGBOOST (2,000 RESAMPLES)**

| Comparison | Queries | ΔAUC [95% CI] | ΔHit@1 [95% CI] | ΔNDCG@5 [95% CI] |
|---|---|---|---|---|
| Five-run averages | All (23,448) | **+0.0004 [+0.0000, +0.0008]\*** | −0.23 pts [−0.44, −0.03]\* | +0.0008 [−0.0001, +0.0017] |
| Five-run averages | New facilities (766) | +0.0031 [−0.0030, +0.0096] | +1.31 pts [−1.31, +3.92] | +0.0100 [−0.0043, +0.0233] |
| Single models | All (23,448) | −0.0001 [−0.0005, +0.0003] | −0.35 pts [−0.52, −0.20]\* | 0.0000 [−0.0008, +0.0008] |
| Single models | New facilities (766) | 0.0000 [−0.0064, +0.0065] | +1.20 pts [−1.02, +3.47] | +0.0083 [−0.0042, +0.0205] |

\* 95% interval excludes zero.

The MC-GNN achieves a significantly higher AUC than XGBoost and is higher in NDCG@5 and Recall@5, though not significantly. XGBoost retains a significantly higher Hit@1, by 0.23 percentage points (0.35 for single models). On facilities with no transfer history, the MC-GNN leads on every metric; with 766 queries these differences are not significant. Averaging five runs benefits the MC-GNN more than XGBoost (AUC +0.0006 against +0.0001).

In a controlled comparison at equal size and number of runs (hidden 64 / output 32, three runs each, same scoring head), the MC-GNN was significantly better than the GAT-only model in AUC and NDCG@5, both overall and for new facilities, and statistically indistinguishable from the GCN-only and GraphSAGE-only models in Hit@1 and NDCG@5. Once the scoring head is pair-aware, much of the predictive signal reaches the model through the shared-neighborhood pooling, and a single strong channel performs comparably to three.

### C. Ablation Study

**TABLE VI — EFFECT OF EACH EXTENSION (EPA TRI 2024 TEST)**

| Variant | AUC | Hit@1 | Hit@1 (new facilities) |
|---|---|---|---|
| MC-GNN, base design (one shared network) | 0.9877 | 96.22% | 72.0% |
| + three networks (multi-view), no channel dropout | 0.9856 | 95.83% | 69.5% |
| + channel dropout | 0.9880 | 96.51% | 77.5% |
| + normalized HSIC, pair-aware scoring | 0.9890 | 96.50% | 75.8% |
| + larger model, average of five runs (final MC-GNN) | **0.9891** | **96.70%** | **76.9%** |

*The first two rows use binary cross-entropy, the eight standard pair features, and up to 500 epochs; later rows add the ranking term, the two network-derived pair features, and up to 1,000 epochs, so part of the change from the second to the third row is due to training and inputs rather than channel dropout alone.*

Assigning separate networks without channel dropout lowered performance, because attention concentrated on a single channel. Channel dropout produced the largest single gain, particularly for facilities without history. A learned per-facility identity embedding was also tested and reduced performance for every model, including XGBoost, consistent with memorization of individual facilities.

### D. Real-Trade Statistics and Platform Results

**TABLE VII — HOW REAL RECOVERY TRADE CHOOSES PARTNERS (EPA TRI, 2020–2022)**

| Measurement | Value |
|---|---|
| Acceptance weight: prior successful transfers | 7.88 |
| Acceptance weight: proximity | 1.82 |
| Acceptance weight: quantity fit | 1.23 |
| Prior transfers alone, AUC | 0.912 |
| Partnerships continuing into the next year | 74.2% |
| Trading pairs new each year | 21.6% |

*Weights from a logistic regression over the same 0–1 encodings the platform computes; under case-control sampling only the slopes carry over, so the intercept is not reported.*

After calibration, partnerships in the generated market continue at 72–75% per round, and the prior-interaction feature separates accepted from rejected platform requests at an AUC of 0.784 (from 0.552). The platform model was retrained on the calibrated data (701 plants, 12,550 exchange requests) and evaluated on the most recent 20% of requests:

**TABLE VIII — PLATFORM DATASET, TEMPORAL HELD-OUT TEST (2,511 REQUESTS)**

| Model | AUC | NDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|---|
| **MC-GNN (deployed)** | **0.8624** | **0.9429** | **0.8074** | **0.8989** |
| Rule-based baseline | 0.7920 | 0.9319 | 0.8031 | 0.8957 |

The retrained model exceeded the previously serving model (AUC 0.8277 on the same split) by more than the 0.02 promotion margin and was promoted. A model trained only on TRI and applied unchanged to the platform data scored an AUC of 0.531 before calibration and 0.720 after, indicating that the calibrated market reproduces patterns found in real trade. The deployed model uses the base MC-GNN design; serving the multi-view, pair-aware model requires building the three networks and pair sets within the recommendation service, which is planned.

### E. Discussion and Limitations

On real industrial transfer data, a multi-channel GNN whose channels read distinct relationship networks, with an HSIC regularizer that acts on training and a scorer that represents shared neighborhoods, outperforms rule-based matching by a wide margin and is competitive with a strong gradient-boosted tree baseline: higher in overall ranking discrimination, marginally lower in top-1 accuracy. The graph models' advantage is that they learn from the relationship networks directly, whereas a tree ensemble can use network information only when it is engineered into tabular features.

Several limitations apply. The TRI data covers chemical recovery transfers in the United States and contains no quality, trust, or rating information. Negative examples are sampled non-partners rather than recorded rejections. Results are reported for a single test year. With a pair-aware head, a single GCN or GraphSAGE channel performs comparably to the three-channel model. The HSIC weight was tuned at the smaller model size and not re-tuned after the final increase in capacity. The platform's market trades are synthetic, calibrated to real statistics, so platform results demonstrate the system rather than real-world accuracy in the Indian context.

---

## VII. Conclusion and Future Work

EcoLinkAI formulates industrial waste-exchange partner recommendation as supervised link prediction with a multi-channel graph neural network. Assigning each channel a distinct relationship network, keeping all channels in use with channel dropout, replacing the inner-product HSIC with its normalized form, and scoring pairs from their shared higher-order neighborhoods produce a model with nearly independent channels. On six years of real EPA TRI recovery transfers, the MC-GNN reached an AUC of 0.9891, NDCG@5 of 0.9754, Recall@5 of 0.9810, and Hit@1 of 96.70% on an unseen test year, against 88.58% for rule-based matching and 76.9% against 48.8% for facilities without history. It significantly exceeded XGBoost in AUC, while XGBoost was 0.23 percentage points more accurate at the top position. Real-trade statistics measured from the same data were used to calibrate the platform's market, on which the deployed model reached an AUC of 0.862 against 0.792 for rule-based matching. Future work will serve the multi-view, pair-aware model in the platform, re-tune the regularizer at the final model size, evaluate across multiple test years, and evaluate on real exchange records from the Indian industrial context as the platform accumulates history.

---

## References

[1] S. Cui, D. Li, and J. Zhang, "MC-GNN: Multi-Channel Graph Neural Networks with Hilbert-Schmidt Independence Criterion," *IEEE Transactions on Big Data*, vol. 11, no. 4, pp. 2036–2045, 2025.

[2] T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," in *Proc. International Conference on Learning Representations (ICLR)*, 2017.

[3] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph Attention Networks," in *Proc. International Conference on Learning Representations (ICLR)*, 2018.

[4] W. L. Hamilton, Z. Ying, and J. Leskovec, "Inductive Representation Learning on Large Graphs," in *Advances in Neural Information Processing Systems*, vol. 30, pp. 1024–1034, 2017.

[5] A. K. Elengovan, N. Seshagiri, O. Owolabi, K. Jangiti, A. K. Padhy, and K. Jayabalan, "Hybrid AI Model for Supply Chain Risk Prediction and Resilience Planning Under Global Disruptions," in *Proc. 2026 International Conference on Innovative Computing, Intelligent Communication and Smart Electrical Systems (ICSES)*, pp. 1–9, 2026.

[6] S. Afrin, F. Ahmed, A. Sattar, and S. A. Mukti, "Graph-Theoretic and GNN-Based Framework for Surplus-Demand Matching in Rooftop Farming Networks," in *Proc. 2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence & Networking (QPAIN)*, pp. 1–6, 2026.

[7] S. Pareek, C. Kalaneya, R. Kumawat, S. Pareek, and Y. Sharma, "Modular Agentic AI Ecosystem for Real-Time Circular Supply Chain Optimization in Oceania's Resource Sector," in *Proc. 2026 IEEE 18th International Conference on Computational Intelligence and Communication Networks (CICN)*, pp. 1742–1747, 2026.

[8] H. Lan, J. Huang, M. Liu, F. Yang, W. Jiang, and D. Niyato, "Graph Convolutional Network-Based Interest Recommendation in the Social Internet of Things with Sparse Social Interaction," *IEEE Internet of Things Journal*, 2026.

[9] H. Li, M. Ge, D. He, and B. Xia, "AI-Based Graph Neural Networks for Predictive Energy Optimization in IoT-Enabled Consumer Electronics," *IEEE Transactions on Consumer Electronics*, 2025.

[10] G. V. Ramteke, V. Gupta, and P. Singla, "Enhancing Recommendation Systems with Graph Contrastive Learning: A Comprehensive Study," in *Proc. 2025 8th International Conference on Trends in Electronics and Informatics (ICOEI)*, pp. 903–908, 2025.

[11] L. Zhang, Z. Chen, W. Zhang, H. Zhao, and L. Wu, "GDiffMAE: Guided Diffusion Enhanced Mask Graph AutoEncoder for Recommendation," *IEEE Transactions on Knowledge and Data Engineering*, vol. 37, no. 12, pp. 7199–7212, Dec. 2025.

[12] J. Min, W. Wei, B. Wang, Z. Gao, and D. Quan, "A Graph Neural Network Recommendation System for Joint Comment Text Representation," *IEEE Transactions on Industrial Informatics*, vol. 22, no. 2, pp. 1028–1037, 2026.

[13] M. Zhao, H. Peng, and L. Li, "Multivariate Time-Series Anomaly Detection Based on Dynamic Graph Neural Networks and Self-Distillation in Industrial Internet of Things," *IEEE Internet of Things Journal*, vol. 12, no. 9, pp. 12181–12192, 2024.

[14] A. Gretton, O. Bousquet, A. J. Smola, and B. Schölkopf, "Measuring Statistical Dependence with Hilbert-Schmidt Norms," in *Proc. International Conference on Algorithmic Learning Theory (ALT)*, pp. 63–77, 2005.

[15] X. Wang, H. Yang, and M. Zhang, "Neural Common Neighbor with Completion for Link Prediction," in *Proc. International Conference on Learning Representations (ICLR)*, 2024.

[16] M. Zhang, P. Li, Y. Xia, K. Wang, and L. Jin, "Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning," in *Advances in Neural Information Processing Systems*, vol. 34, 2021.

[17] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, "Similarity of Neural Network Representations Revisited," in *Proc. International Conference on Machine Learning (ICML)*, pp. 3519–3529, 2019.

[18] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, 2016.

[19] U.S. Environmental Protection Agency, "Toxics Release Inventory (TRI) Basic Plus Data Files," 2019–2024. [Online]. Available: https://www.epa.gov/toxics-release-inventory-tri-program
