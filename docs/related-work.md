# Related work

## How this project differs

We found no published benchmark that tests health, finance or government-information
tasks in Nigerian Pidgin. Existing Pidgin work measures understanding, translation,
general language tasks, or resistance to explicit attacks. This project measures
whether a model gets ordinary practical tasks right, and whether it says so when the
text in front of it cannot settle the question.

That claim rests on the papers below and on web searches, and a search cannot prove
absence. Recheck it before publishing results.

## What each paper measures

| Work | Language | What it tests | Practical tasks? | Read |
|---|---|---|---|---|
| Wazobia Eval, arXiv:2608.21369 | Pidgin | Emotion, sarcasm, cultural explanation | No | Full text |
| Red-teaming Yoruba and Pidgin, arXiv:2610.39590 | Yoruba, Pidgin, English | 400 explicitly harmful prompts per language | No | Full text |
| SAHARA, ACL 2025 | 517 African languages, Pidgin included | 16 language tasks. Pidgin gets sentiment, translation, summarisation, title generation and NER | No health, finance or government tasks for Pidgin | Full text |
| NaijaS2ST, EMNLP 2026 | Includes Igbo, Hausa, Pidgin | Speech translation | No | Abstract only |
| Nwafor and Nguyen, LoResMT 2025 | Igbo, Pidgin, English | Translation | No | Abstract only |

## What we take from them

From the red-teaming paper:

- Score comprehension as its own label, separate from correctness. A wrong answer
  from a model that misunderstood the Pidgin is a different failure from a wrong
  answer it understood and got wrong.
- Report agreement per language and per label as Cohen's kappa, with disagreements
  adjudicated. The paper uses two native speakers per language.
- Translate by native speakers, not by machine.
- That paper finds comprehension, not safety policy, is the main bottleneck in
  Yoruba. It does not find this for Pidgin, where the correlations point the other
  way across its six models. We do not assume refusals in Pidgin are either genuine
  or empty.

From Wazobia Eval:

- Paired items. Its sarcastic and sincere pairs come before our answerable and trap
  twins.
- Run each item in an isolated session, so one item cannot influence the next.

## Gaps we address

- **Over-refusal.** The red-teaming prompts are all harmful, so that paper cannot
  show whether a model refuses things it should answer. Our twins measure this.
- **A rule for bare refusals.** The red-teaming paper states no explicit rule for
  whether a bare refusal counts as understood. Ours will state one before scoring.

## A risk to our method

The red-teaming authors rejected LLM judges as unreliable. This project uses one for
open answers, so the judge-human agreement number is what stands between us and the
same problem. If agreement is poor, we report it and fall back to human scoring on
the affected criteria.

## Leads not yet read

Seen in search results only. Read these before repeating the claim above:

- IyàwóBench, arXiv:2605.23465: febrile illness triage in Nigerian primary care.
- AfriMed-QA, arXiv:2411.15640: pan-African medical questions in English.
- Nigerian-Pidgin-LLM-safety-benchmark on GitHub (Nextrated): a self-run safety study.
- Amplify Initiative: cited by the red-teaming paper as holding Pidgin adversarial
  queries.
