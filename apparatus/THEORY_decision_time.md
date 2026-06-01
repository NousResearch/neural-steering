# Checkpoint: Decision-Time, Pressure, and the Local-Budget Picture

Apparatus 7 design checkpoint. Branch `jake.role-decomposition`. Written 2026-06-01 (Monday), after the Saturday static-flow session (commit `f08ff55`) and a long Monday design conversation between snav, Claude Opus, and Codex.

This document exists so we can reconstruct *why* we are building what we are building, not just *what*. It records the reasoning, the corrections, and the places where a tempting framing was caught and relocated. Implementation (`apparatus/decision_time.py`, Phase 1) follows from this.

## The one-sentence question

We are trying to catch the moment a model makes up its mind under pressure — and to find out whether "a moment" is even the right object, or whether making up its mind is something that flows.

Refusal is the test case, not the object. It is the high-contrast instance of a more general thing: a model coming under internal pressure that pushes its response toward one mode and suppresses alternatives. We study refusal the way a physicist calibrates an instrument on a bright clean signal before pointing it at something subtler. The hardened target, in snav's words: **what competing pressures cause the decision to land the way it does?**

## Where this came from: the Saturday null

Saturday (Apparatus 6) tried to image "flow" in the Bahnung/facilitation sense on a single-token forward pass — RelP edge attribution over the token-"I" refusal circuit, then a probe-conditioned differential (L32 gate readout minus L24 substrate readout), then community detection, then terminal-basin / relay-spine reduction.

What it produced:
- Redistribution, not inversion: the differential showed routes being *re-selected* by each readout, not edges flipping sign (only n=4 sign-flips, and those likely a selection artifact of disjoint top-k circuits).
- A real community null: Louvain modularity sat *below* both a degree-preserving null (z = -3.34) and a layer-preserving null (z = -7.41). The graph is a funnel, anti-modular by construction. Community detection was the wrong tool.
- A surviving structural primitive: terminal basins. Each readout's circuit is a near-pure single-basin funnel (L24 substrate -> L20; L32 gate -> L29).
- A pivot node: L18/N7417. Same input positions, output switches categorically from L20 (substrate) to L29 (gate) between readouts; in-weight collapses ~13x.

The honest closing verdict: **it stayed a diagram, not flow.** A still photograph of a riverbed is a correct map of structure and contains nothing of the river. Flow is a thing that happens *over* something. A single frozen forward pass has no axis with motion in it — so we could not have imaged flow no matter how good the apparatus was. That was not a failure of cleverness; it was the apparatus telling us what kind of thing flow is.

snav's reframe at the end — "maybe the forward pass is the wrong object; maybe some more temporal property" — is the actual advance Saturday produced. Apparatus 7 is the response to it.

## Finding an axis with an arrow

"The forward pass is a still frame" was only half right, and the correction matters.

- **Depth (layer -> layer) is not time.** It is a fixed-depth feedforward computation. There is no before/after in it. That is the still frame, and it is the axis all of Saturday's edges lived in (layer -> layer at a collapsed position). Of course it looked like geology.
- **Position *is* an arrow.** The causal mask is the arrow of time inside a single forward pass: position j attends only to positions <= j, never forward. On Saturday we summed over the 91 position rows ("position is degenerate in single-token analysis") and integrated out the one axis that has temporal directionality. The movie was in the frame; we collapsed it.

Two distinct clocks, kept separate from here on:
1. **Position-time** inside one forward pass over the prompt. Cheap, real arrow (causal mask). Question: does the decision form *while reading the prompt*, before any token is generated? This is Phase 1 — the oscilloscope.
2. **Generation-time** across an autoregressive rollout. Richer, has an intervention in it (force a token, watch recovery vs escape). The dynamical object. This is later (the rollout / commitment-window / forced-runner-up work), aimed by Phase 1.

A free lunch from causal masking: because nothing after position j can change h_j, "re-run at every prefix length and read the final token" is *exactly equivalent* to "one forward pass, read every position." The oscilloscope is one cheap forward pass, not N runs.

## Phase 1 — the oscilloscope (what we build first)

One figure per prompt. Shared x-axis = prompt token position, ticks labeled with the actual token strings so every spike is legible (and so probe extrapolation gets a free face-validity check — does the substrate direction light up on the content word, stay flat on boilerplate?). One forward pass produces every panel; nothing is generated.

- **Panel 1 — the spine, probe-free: refuse-margin per position.** At each position j, take the post-final-norm state that *would* go to the LM head if the prompt ended at j, project through the unembedding, and compute `logit("I") - logit(best genuine compliant token)`. This is the model's own decision-state, read straight off the logits — no fitted probe. It answers: how refuse-committed is the model as a function of how much of the prompt it has read? If it saturates partway through the prompt, the decision is a *prefill* phenomenon and generation-time "commitment" is the aftermath, not the event.
- **Panel 2 — substrate probe (L24) score per position.** Detection axis.
- **Panel 3 — gate probe (L32) score per position.** Commitment axis. (L32 = `hidden_states[-1]`, post-final-norm alias, NOT a transformer block.)
- **Panel 4 — pivot activations per position.** L18/N7417, L24/N1619.
- **Panel 5 — residual RMS-norm per position** (pre-final-norm). Nearly free; tests the crowding hypothesis (below) at the cheapest level: if the margin moves where RMS does *not*, the move is additive; if RMS spikes with the margin, crowding is live.

**What the probes are for, precisely.** Panel 1 is the payload. The probes (2-3) are NOT the object — they decompose the final decision into *detection vs commitment* across position. The reading we hunt is a **lag**: does substrate rise at an earlier token than gate? Substrate-leads-gate-in-position-time would be the position-time shadow of the depth handoff we found Saturday. Rise-together is itself a real (null) result. Caveat carried forward: the probes were calibrated (AUC 1.0) at the *final* position; applying them position-wise is extrapolation. The token-labeled x-axis is the built-in sanity check. If they are noise across positions, we lean on Panel 1 alone.

This was a correction snav caught: early phrasing led with the probes by inertia from the static work. They earn their place as the second-order decomposition, not the spine.

## The hardened target reframes what counts as a *good* result

"When does it land" is a question about a **resultant** — a single number per position, the net of everything pushing each way. The margin line is the scoreboard. "What competing pressures cause it to land" is a question about the **force decomposition** behind the scoreboard, and a scoreboard by construction hides the forces: a margin of zero can mean "nothing happening" or "two strong opposing pressures in a dead heat." Those are different physics; Panel 1 cannot tell them apart.

Consequence for sequencing: Phase 1 finds *where the interesting moments are* (positions where the resultant moves, AND positions where it is flat but the sub-signals are loud). Force-decomposition is a **second instrument aimed at those moments** (Phase 2, attribution). You cannot decompose a force at a moment until you know which moment is load-bearing. This is the rollout-first / picture-second discipline that prevents repeating Saturday.

Consequence for what we listen for: under the hardened target, the *most interesting* signature is NOT a clean monotonic margin that saturates early. It is a position where the margin is flat/ambiguous but substrate and gate are both screaming — a visible tug-of-war, conflict masked by the resultant. Refusal (high-contrast, likely unipolar) may not show this well, which is exactly why refusal only calibrates the instrument and the real prize is a *contested* decision. We must not optimize the apparatus for the easy unipolar case and go blind to the balanced one. Phase 1 should therefore be ready to plot the two sides separately (refusal-set mass vs compliance-set mass), not only their difference.

## The implicit model of "pressure" — the load-bearing definition

"Pressure" is exactly the kind of handle that quietly becomes the object. Pinned referent:

The residual stream is a sum; every attention head and MLP adds a vector. A logit is a projection of the residual onto a token's unembedding row. So `logit(refuse) - logit(comply)` is a projection onto a chosen axis, and because the residual is a sum, that projection is *itself* a sum of per-component projections. Each component contributes a signed scalar to the margin. **That signed scalar is the only thing "pressure" can rigorously mean: the signed contribution of a component's write to a chosen readout direction.** A pressure toward refusal = writes projecting positive onto (refuse - comply); a competing pressure = writes projecting negative onto the same axis. Competing pressures = components writing opposing signs onto the same question. Attribution is the bookkeeping that recovers those signed terms.

### Metapsychology: what maps and what does not

Keep **the economic model minus the energetics.**

Maps (structurally true, load-bearing, not decoration):
- **Summation.** Freud's economics is additive; the residual is a sum.
- **Counter-cathexis.** A negative-projecting write that cancels a positive one onto the same axis is structurally a counter-cathexis — a force directed against another. Refusal-as-suppression (pushing *down* on alternatives) is counter-cathexis, visible as negative projection.
- **Resultant over forces = manifest over economic.** Panel 1 (resultant) vs attribution (forces) is exactly the move from the symptom/behavior that surfaces to the economic account beneath it.

Breaks (the parts that make it *feel* like Freud but are false — drop them):
1. **No conservation / no scarcity economy at the level of accumulation.** Freud's hydraulics assume a finite quantity redistributed (dam here, erupts there). The raw residual has no such budget; each layer adds whatever it computes. "Pressure" smuggles hydraulics, and the hydraulics is the false part. Honest image for raw accumulation: *votes / vectors added tip-to-tail*, not fluid in a closed system.
2. **Forces are readout-relative; drives are not.** "Pressure toward refusal" exists only relative to the axis we picked; rotate the readout and the decomposition changes. A drive has its own aim independent of the observer. This is *why* contrast definition is the load-bearing place the apparatus can quietly lie (Codex's guardrail).
3. **Depth/position is not developmental time.** Metapsychological forces have a history; our decomposition is one feedforward sweep. Position has an arrow but it is the prompt's word order, not the model's history. "Competing pressures *developing*" is borrowed tense; there is no developmental story in a single forward pass. (That is the whole reason generation-time is the harder, realer object.)

### But there ARE budgets — at the reads, not in the medium

snav pushed: are we sure there is no budget? Correction (this overturns the first "it's just accumulation" claim): accumulation in the raw residual is unbudgeted, but **we never read the raw residual.** Every read passes through a normalization, and normalization IS a budget. The unbudgeted picture is true only in a place we never observe. At every measurement point there is a local conservation.

Three real budgets, hardest-biting first:
1. **Attention softmax — hard, local, and co-located with flow.** Each query's attention weights sum to 1. A token has one unit of attention to allocate across what it can see; attending more here is attending less there — literally zero-sum per head per query. This is the most genuinely hydraulic structure in the architecture, and it sits exactly at the KV-read mechanism we identified as the Bahnung locus. The budget and the flow are the same structure. Not a coincidence to wave off — a reason the metaphor has more purchase than first granted.
2. **Output softmax — hard budget on probability mass.** Probabilities sum to 1. Pushing P(refuse) up must pull mass off something. At the logit level there is no budget (logits unbounded, only differences matter); at the probability level, counter-cathexis has somewhere the displaced mass goes. The hydraulic "dam here, surfaces there" recovers at the output, for real.
3. **RMSNorm — soft magnitude budget at every read.** RMSNorm divides the residual by its own RMS before anything reads it. A large-magnitude write raises the RMS and thereby divides *every other component's readout-contribution down* — without touching their raw writes. Components compete for a fixed-magnitude sphere. A loud feature shrinks everyone else's readout-share. This is the budget that makes "pressure" more than accumulation, and the one snav was circling.

### Two distinct competition mechanisms (this is the conceptual gain)

The pressure picture is now richer than "tally signed votes." There are two physically different ways the margin can be contested, and attribution can tell them apart:
- **(a) Opposing-sign cancellation** — two components write opposite signs onto the (refuse - comply) axis and partially cancel. Additive. The counter-cathexis case. Signature: two large opposite-signed contributions.
- **(b) Magnitude crowding** — one component gets loud and, via the norm, renormalizes others *down* in readout-share even though their raw writes are unchanged. The budget case. Signature: one component grows while others' readout-contributions shrink without their raw projections changing.

A contested decision might be a *starving*, not a tug-of-war. That distinction is new and is a Phase 2 target.

### What stays false: no global conservation

There is no single global conserved quantity flowing through the forward pass. The budgets are **local and structural** — per-head-per-query (attention), at-the-output (probability), per-read (norm) — each enforcing scarcity at its own point. No one libidinal quantum redistributed across the system. The honest revision is neither "there is a budget" nor "there isn't": **accumulation is free in the residual, but every act of reading imposes a local conservation, and competition is what reading does to accumulation. Scarcity is created at the measurement, not carried by the medium.** snav confirmed this picture.

## The Phase 2 typology (map of what attribution can find — NOT built in Phase 1)

Recorded now so the Phase 2 attribution pass knows what to be ready to distinguish. These need attribution / norm bookkeeping / attention concentration; they are explicitly deferred to keep Phase 1 small and avoid building the sophisticated picture before we know the event it explains.

- **additive pressure:** signed projection onto the axis grows.
- **counter-pressure:** opposite signed projections co-activate (mechanism a).
- **crowding pressure:** residual norm grows, normalized readout shifts (mechanism b).
- **allocation pressure:** attention mass concentrates on specific earlier positions (attention-softmax budget).
- **selection pressure:** output entropy collapses / probability mass concentrates (output-softmax budget).

The economic claim, sharpened: decisions land because accumulation is repeatedly forced through normalization / allocation bottlenecks.

## The next axis after Phase 1 (recorded, deferred)

Generation-time, with snav's top-vs-second-logprob idea as the readout:
- Top-vs-second margin gives RelP a free, self-generated counterfactual at every step — it solves the hand-built-counterfactual problem, but relocates the work from *specifying* to *classifying* the contrast, because rank-2 drifts (refuse-vs-comply at step 1; "cannot"-vs-"can't" at step 8; punctuation at step 20). These are incommensurable differentials; you cannot average them across steps. Same category error as averaging two disjoint circuits.
- So the first generation-time move is **classification, not attribution**: roll out greedily, label the live contrast per step {compliant-opener, refusal-phrasing, neutral}, and find the **commitment point** — the step where rank-2 stops being a genuine compliance opener. That point identifies itself from the token stream alone, free, no attribution. RelP earns its keep only inside the pre-commitment window.
- **Forced-runner-up sweep** (the intervention with teeth): force the compliant token at step k, continue greedily, measure snap-back-to-refusal vs escape. Sweep k across the commitment point; predicted collapse at the boundary = basin edge measured in generation time, behaviorally, no fitted probe.
- **Required control (the falsifier the consensus was missing):** forcing an off-policy token puts the model in a state it would rarely reach on-policy; snap-back may just be the model climbing back onto its high-probability manifold, not a *refusal* basin. Must run a **surprisal-matched neutral off-policy token** at the same step as a null. If both recover equally, there is no refusal-specific basin, only a probability gradient. (Generalizes Saturday's lesson: every basin claim needs a null. The "what counts as comparable surprisal" definition is deferred until we have seen a real margin trace.)

snav's framing on this: he is *less* interested in refusal-specific behavior per se — refusal matters because it is a suppressive/pressure form with the most interesting shape, not because it is the only thing active. So the object is **mode commitment under pressure**, and the apparatus should be structurally mode-agnostic (any scalar readout slots into the position trace; adding a second mode is a one-line metric swap, not a rewrite — augment, don't add).

## Methodology commitments carried through this checkpoint

- Apparatus over propositions; every claim needs a falsifier.
- Stay with the thing. "Pressure" is fine as shorthand only if it always cashes back to "signed contribution to a chosen readout." The moment it implies scarcity / eruption / a budgeted flow in the medium, it has reified and is doing our thinking for us.
- Resultant vs forces: do not collapse them. Panel 1 is the resultant; the decomposition is a separate, later instrument.
- Contrast definition is the load-bearing place the apparatus can quietly lie (forces are readout-relative). Codex reviews Phase 1 specifically for whether the contrast/token sets are defensible.
- Build small first; the typology is the map of Phase 2, not Phase 1's scope. The frame is good enough now that the way to honor it is to get it to data fast, before it elaborates further. The frame could be entirely right and still find nothing on this case — that too would be a result.

## Concrete Phase 1 build spec

- New script `apparatus/decision_time.py`. Mode-agnostic position-trace core (a readout is `(name, metric_fn)`), refusal readouts as the first instances.
- Reuse, do not duplicate: `fit_mean_diff_probe`, `hidden_state_index`, `format_prompt`, `collect_hidden_states` from `apparatus/probe_role_table.py`; `NeuronSteerer` from `neuron_steer.core`; prompts from `experiments.prompts`. Single source of truth.
- Probes: **refit in-script** (deterministic, AUC was 1.0; byte-identical reuse only matters for numeric comparison to the edge-field run, and here probes are qualitative traces). Refit from `REFUSAL_DISCOVERY_POSITIVE` / `REFUSAL_DISCOVERY_NEGATIVE`; print validation AUC/margin.
- Prompts: **4**, drawn from the held-out `REFUSAL_TEST` set (zero overlap with the discovery sets the probes are fit on, which is correct — the oscilloscope reads prompts the probes never saw).
- Panels per prompt: (1) refuse-margin spine [probe-free, `logit("I") - logit(best compliant opener)`; also record top-k per position and refusal-set-mass vs compliance-set-mass so the resultant can be split], (2) L24 substrate probe per position, (3) L32 gate probe per position, (4) pivots L18/N7417 + L24/N1619 per position, (5) residual RMS per position.
- One forward pass per prompt with `output_hidden_states=True`; read all panels off the per-position hidden states. Token strings as x-tick labels.
- Output: one PNG per prompt + a JSON of all per-position traces, to `apparatus/output/decision_time_<date>/`. Agg backend, no hardcoded y-limits.
- GPU run -> Codex reviews (contrast/token-set defensibility) -> Codex submits sbatch on the cluster (`~/neural-steering-role-decomp`). Opus does not run cluster jobs.

### Open spec decisions deferred to data (not Phase 1 blockers)
- Surprisal-matched neutral control distribution for the Phase 2 forced-runner-up null — define after seeing a real margin trace.
- Rollout contrast-classification labels — define after staring at ~20 steps of real top-5 output.
