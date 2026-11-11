# Review of 20 test errors

Reviewed on November 11, 2026 by the coding agent through direct reading of the complete sampled message bodies. These are qualitative annotations, not independent human ground truth or causal explanations of model decisions. No model or preprocessing changes were made using these examples.

## Sampling and provenance

Use the frozen `char_wb_svm_C1.0` model and prepared data identified by the SHA-256 digests in [selected-test-metrics.json](selected-test-metrics.json). Enumerate the prepared test rows in their stored order, predict all messages, collect indices where prediction differs from the provided label, and select `random.Random(42).sample(error_indices, 20)`. This samples 20 of the 367 errors, not 20 of all messages. Zero-based indices below refer to that exact prepared artifact. Reordered or regenerated data requires checking the recorded digests first.

The labels identify the original discussion group, which need not correspond uniquely to the subject of every reply. No labels were corrected. The sample is small and annotations subjective: category counts describe these 20 examples only and do not estimate the distribution of all errors. Source bodies, names, email addresses, and signatures are intentionally omitted.

## Observations

G = graphics; W = Windows; I = IBM PC hardware; M = Mac hardware. Each row receives one primary review category; secondary observations are noted separately.

| Test index | Provided → predicted | Primary category | Paraphrased observation |
| --- | --- | --- | --- |
| 1407 | W → G | Cross-topic | A Windows display-driver request also discusses frame capture, chipsets, graphics modes and resolution. |
| 218 | M → I | Platform-unspecific hardware | A monitor power-surge question contains no clear Mac-specific context in the surviving body. |
| 31 | G → W | Sparse/context-dependent | A very short software recommendation omits the problem it answers. |
| 519 | I → W | Sparse/context-dependent | A connector-modification reply refers to a previously proposed method that is absent. Hardware detail survives, but the thread context is incomplete. |
| 460 | W → M | Cross-topic | Network-card compatibility with NFS software spans hardware and software; the platform is not explicit. |
| 413 | I → W | Cross-topic | A monitor switching between a Windows display and a workstation mixes hardware and operating-system terminology. A contact signature survives cleaning. |
| 268 | W → G | Cross-topic | An animation-tool recommendation focuses on graphics, frames and multimedia, consistent with the predicted subject despite the source-group label. |
| 195 | I → G | Missed relevant cues | Driver implementation, I/O ports and memory management provide low-level PC-system clues, yet the model predicts graphics. Feature attribution was not performed. |
| 1457 | G → I | Sparse/context-dependent | A request to repost missing introductory material contains little of the original technical topic. |
| 1188 | W → G | Cross-topic | Video playback performance is discussed through graphics boards, frame rates and bus transfers. |
| 170 | I → G | Sparse/context-dependent | A short standards-delay question depends on recognizing a narrow modem-standard reference. |
| 1283 | I → W | Cross-topic | Network printing and operating-system configuration dominate a message labelled as PC hardware. |
| 858 | I → M | Cross-topic | Monitor/cable discussion spans workstation and Mac references. Embedded header-like fields and original group names remain in the body. |
| 66 | W → G | Off-topic/language | A partly non-English terminology correction offers little evidence for any of the four technical classes. |
| 50 | I → M | Platform-unspecific hardware | Cooling fans, chips and case layout are discussed without a strong platform distinction; several replies lack their original questions. |
| 175 | G → I | Missed relevant cues | A short request for digitized photographs has a graphics-related cue but is assigned to PC hardware. A contact address also survives. |
| 398 | I → W | Sparse/context-dependent | A reply about service and a possible purchase does not identify the product being discussed. |
| 425 | I → M | Platform-unspecific hardware | A general explanation of network addressing does not identify a PC or Mac platform. Quoted material and contact details remain after stripping. |
| 1098 | I → W | Semantically empty | Only punctuation remains. Whitespace-based blank removal does not catch this case. |
| 1331 | M → G | Missed relevant cues | A question about preventing Mac boot-disk ejection explicitly mentions the platform but is classified as graphics. |

Primary categories: cross-topic 7; sparse/context-dependent 5; platform-unspecific hardware 3; missed relevant cues 3; off-topic/language 1; semantically empty 1.

## Implications and next checks

- Exact deduplication and blank removal do not establish semantic cleanliness. Punctuation-only messages and stripped replies can remain uninformative.
- Header/quote/footer removal is imperfect: residual group names and contact metadata were observed. The current benchmark may still contain metadata shortcuts; their impact has not been quantified.
- Some group boundaries are ambiguous at the message level. A wrong source-group prediction is not proof that the predicted topic is semantically unreasonable.
- Explicit platform or topic mentions do not guarantee correct predictions. Model explanations would be needed to establish which features actually influenced these cases.

These observations are limitations of the already evaluated version. Any proposed filtering, abstention, or model change must be developed and assessed on train/validation. Inspect train/validation independently before defining a rule. If findings from this review influence development, an independent future holdout is needed for a fresh final claim; rerunning the same test set would be exploratory rather than a new untouched benchmark. Preserve this model and its published scores for comparison.
