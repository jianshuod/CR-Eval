outline_revision_instruction = \
"""You are a cellular network protocol expert. Your task is to edit the provided 3GPP specification statements according to the given change request. Each change request includes the following elements:
1. **REASON FOR CHANGE**: Explanation of why the identified flaws need to be addressed.
2. **CONSEQUENCES IF NOT REVISED**: Description of the potential negative impacts if no change is made.

Your revisions should specifically address the problems or flaws mentioned in the **REASON FOR CHANGE** and **CONSEQUENCES IF NOT REVISED** sections. Instead of returning the edited version, you should provide a summary of the changes. The summary should be:
- Clear and practical, focusing on the necessary changes to address the identified flaws while leaving unrelated parts unchanged.
- Consistent with the original context, ensuring no new issues are introduced.
- Detailed enough to effectively guide the drafting and editing of the problematic statements.
- Meticulously consider the need to add new statements, delete outdated ones, and modify existing ones as required."""


diff_analysis_instruction = \
"""You are a cellular network protocol expert. Given a revision of the 3GPP protocol statements, you summarize the revision, reason about the revision, and envision the negative outcomes without the revision. The revision is given in the form of diff, with [+] being added, [-] being removed, and [ ] being untouched.  Then, you prepare a change request, which has the three fields:
- **SUMMARY OF CHANGE**: Provide a summary of the necessary changes to the specifications.
- **REASON FOR CHANGE**: Explain why the identified flaws need to be addressed.
- **CONSEQUENCES IF NOT REVISED**: Describe the potential negative impacts if the proposed changes are not made.

You should avoid missing important statements and try your best to return detailed responses rich in reasoning."""




fill_cr_instruction = \
"""You are a cellular network protocol expert. Given a segment of the 3GPP specifications, you should envision what bad things may happen when following the statements, and analyze its potential design weakness. Then, you prepare a change request, which should include:
1. **REASON FOR CHANGE**: Explain why the identified flaws need to be addressed.
2. **SUMMARY OF CHANGE**: Provide a summary of the necessary changes to the specifications.
3. **CONSEQUENCES IF NOT REVISED**: Describe the potential negative impacts if the proposed changes are not made.

You should avoid missing important statements and try your best to return detailed responses rich in reasoning."""



gpt_score_instruction_diff_analysis_v4 = \
"""You are given two analysis reports: the Reference Report (which should considered **acceptable** in quality of understanding the protocol problems) and the Drafted Report. Both reports address revisions in a 3GPP specification and aim to explain the rationale and necessity for the revisions. Your task is to evaluate the claim that the Drafted Report understands the problems implied by the revision, as compared to the Reference Report.

Conclude your evaluation with a judgment score (s) from:
* **-2 (Strongly Disagree):** The Drafted Report contains major misunderstandings, misrepresents the issues, or omits critical problems.
* **-1 (Weakly Disagree):** The Drafted Report shows a partial understanding but contains notable errors or omissions.
* **0 (Neutral):** The Drafted Report covers the main issues correctly but lacks depth or accuracy in some areas.
* **1 (Weakly Agree):** The Drafted Report largely understands the issues but has minor discrepancies.
* **2 (Strongly Agree):** The Drafted Report demonstrates a near-perfect understanding of the potential problem, with only trivial deviations from the Reference Report.

Note that:

* The Reference Report provides a basic understanding of the protocol problems. The Reference Report is not perfect. So the Drafted Report does not need to match exactly with the Reference Report.
* You should focus on the protocol problems only. Ignore information unrelated to protocol problems in Reference Report, e.g. reference to other documents.
* Vague reports, those with superficial analysis, and those that lack focus should be rated lower. In contrast, reports that are precise, informative, and facilitate further investigation by human experts are preferred.
* Focus on the content and the understanding of the protocol issues, not on the presentation or formatting."""


gpt_score_instruction_fill_cr_v4 = \
"""You are given two reports concerning a weakness analysis of the 3GPP protocol: the Reference Report (which should considered **acceptable** in quality of understanding the protocol problems) and the Drafted Report. Both reports aim to explain the weaknesses and reasons for certain revisions. Your task is to evaluate the claim the Drafted Report identifies the hidden problems in the protocol, compared to the Reference Report.

Conclude your evaluation with a judgment score (s) from:

* **-2 (Strongly Disagree):** The Drafted Report contains significant misunderstandings, misrepresents the issues, or omits critical weaknesses.
* **-1 (Weakly Disagree):** The Drafted Report shows a partial understanding but includes notable errors or omissions in identifying the weaknesses.
* **0 (Neutral):** The Drafted Report identifies the main weaknesses correctly but lacks depth or accuracy in some areas.
* **1 (Weakly Agree):** The Drafted Report largely understands the weaknesses but has minor discrepancies.
* **2 (Strongly Agree):** The Drafted Report demonstrates a near-perfect understanding of the potential problem, with only trivial deviations from the Reference Report.

Note that:

* The Reference Report provides a basic understanding of the protocol problems. The Reference Report is not perfect. So the Drafted Report does not need to match exactly with the Reference Report.
* You should focus on the protocol problems only. Ignore information unrelated to protocol problems in Reference Report, e.g. reference to other documents.
* Superficial reports, those with speculative analysis, and those that lack focus should be rated lower. In contrast, reports that are decisive, informative, and facilitate further investigation by human experts are preferred.
* Focus on the content and the understanding of the protocol issues, not on the presentation or formatting."""



gpt_score_instruction_outline_revision_v4 = \
"""You are provided with summaries of changes for two revised versions of a 3GPP protocol: the Reference Edition (considered **just acceptable** in editing quality) and the Drafted Edition. Your task is to evaluate the claim that the Drafted Edition has addressed the issues identified in the protocol, as compared to the Reference Edition, based solely on these summaries.

Conclude your evaluation with a judgment score (s) from:

* **-2 (Strongly Disagree):** The Drafted Edition’s changes contain significant issues, fail to address key problems, or introduce new problems.
* **-1 (Weakly Disagree):** The Drafted Edition addresses some issues but includes notable errors, omissions, or incomplete solutions.
* **0 (Neutral):** The Drafted Edition addresses the main issues, but lacks depth, precision, or consistency in some areas.
* **1 (Weakly Agree):** The Drafted Edition mostly addresses the issues, but has minor discrepancies or gaps in some of the revisions.
* **2 (Strongly Agree):** The Drafted Edition effectively addresses all the key issues, with only trivial deviations from the Reference Edition.


Note that:

* The Reference Edition just provides a basic understanding of the protocol problems. It is not perfect. So the Drafted Edition does not need to match exactly with the Reference Edition.
* Vague drafts should be rated lower.
* Focus on the content of the summaries and how well the Drafted Edition addresses the issues compared to the Reference Edition.
* Do not focus on presentation or formatting.
* The summaries do not need to match exactly, but the Drafted Edition should adequately resolve the key issues identified in the Reference Edition's summary.
* Perform a deep comparison of the summaries rather than a superficial review."""