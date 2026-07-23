# FlagCX Project Governance

FlagCX is an open-source project governed by its maintainers with the support of the
Beijing Academy of Artificial Intelligence (BAAI). This document describes the
project's roles, decision-making process, maintainer changes, and escalation process.

## Principles

FlagCX aims to make technical decisions openly and on their merits. Project
participants are expected to:

- discuss changes constructively and in good faith;
- consider the long-term interests of the project and its users;
- give community members a reasonable opportunity to participate in decisions; and
- follow the [Code of Conduct](CODE_OF_CONDUCT.md) and
  [Contributing Guide](CONTRIBUTING.md).

## Project Roles

### Contributors

A contributor is anyone who contributes to FlagCX. Contributions include code,
documentation, issue reports, design proposals, pull-request reviews, testing, and
participation in project discussions.

Contributors may:

- open and participate in issues and pull requests;
- propose technical, documentation, and governance changes; and
- review and test proposed changes.

### Maintainers

Maintainers are responsible for the ongoing stewardship of FlagCX. The current
maintainers are listed in [MAINTAINERS.md](MAINTAINERS.md).

Maintainers are expected to:

- review issues and pull requests;
- guide the project's technical direction and development priorities;
- help contributors participate effectively;
- keep the project reliable, secure, and maintainable;
- enforce the Code of Conduct; and
- participate in project decisions and votes.

Maintainers have permission to approve and merge changes and to perform other
repository administration needed for the project.

### Code Owners

Code ownership is a responsibility assigned to maintainers for particular parts of the
repository; it is not a separate community role. Code owners provide subject-matter
review for changes in their areas. The current assignments are recorded in
[`.github/CODEOWNERS`](.github/CODEOWNERS), which is the source of truth for code
ownership.

### Technical Lead

BAAI selects one of the maintainers to serve as technical lead. The technical lead
helps coordinate the project's overall technical direction and resolves tied
maintainer votes as described below. The current technical lead is
[`MC952-arch`](https://github.com/MC952-arch).

The technical lead does not unilaterally decide ordinary project matters when
maintainer consensus can be reached.

## Decision-Making

Project decisions should normally be discussed in the relevant public GitHub issue or
pull request. Design rationale, significant concerns, and the resulting decision
should be recorded there so that contributors can understand and participate in the
process.

### Consensus

FlagCX prefers consensus over formal voting. Consensus means that the relevant
participants have had a reasonable opportunity to comment, material concerns have
been considered, and no maintainer has an unresolved objection.

Routine decisions may be made through the normal issue and pull-request process.
Broader or potentially incompatible changes should first be proposed in an issue so
their impact can be discussed before implementation.

### Pull-Request Approval

A pull request normally requires at least two approvals before it is merged. At least
one approval must come from a code owner for the affected area.

In an exceptional and time-sensitive circumstance, a maintainer may merge a pull
request without satisfying the normal approval requirement. This should be rare. The
maintainer must explain the reason for the exception in the pull request, and the
change remains subject to follow-up review and correction.

### Voting and Escalation

If maintainers cannot reach consensus after reasonable discussion, any maintainer may
call for a vote in the relevant issue or pull request. All active maintainers must be
given a reasonable opportunity to vote.

Each active maintainer has one vote. A proposal passes when it receives more than 50%
of the votes cast. Abstentions are not counted as votes cast. If the vote is tied, the
technical lead decides the outcome and records the decision and its rationale in the
same public discussion.

An active maintainer is one who has participated in at least one pull-request review,
code contribution, or project meeting during the preceding six months.

## Maintainer Changes

Maintainer status recognizes sustained responsibility for the project; it is not
granted solely on the basis of employment or organizational affiliation.

Existing maintainers may recommend a contributor for maintainership based on the
quality and continuity of the contributor's work, technical judgment, collaboration,
and demonstrated commitment to FlagCX. BAAI has final authority to approve maintainer
appointments.

A maintainer may step down at any time by notifying the other maintainers and updating
[MAINTAINERS.md](MAINTAINERS.md).

A maintainer who has performed none of the activities described above for six
consecutive months is automatically considered inactive. Inactive maintainers are not
counted as active maintainers for voting purposes. Participation in an eligible
activity restores active status unless the maintainer has formally stepped down or
been removed.

BAAI has final authority over the removal of maintainers. Removal may be considered
for prolonged inactivity, serious or repeated violations of the Code of Conduct, abuse
of project privileges, or conduct that materially harms the project. Except where
urgent action is required for security, safety, or legal reasons, the maintainer should
be informed of the concern and given an opportunity to respond.

Changes to the maintainer list must be recorded in
[MAINTAINERS.md](MAINTAINERS.md).

## BAAI's Role and Reserved Authority

The maintainers govern FlagCX's ordinary technical and community work through the
processes above. As the organization backing the project, BAAI retains final authority
over critical project decisions, including:

- appointment and removal of maintainers;
- selection of the technical lead;
- transfer, dissolution, or fundamental reorganization of the project;
- licensing and intellectual-property matters;
- project names and trademarks;
- significant security, legal, financial, or compliance matters;
- major changes to the project's scope; and
- changes to BAAI's reserved authority under this document.

Maintainers should discuss critical decisions publicly when confidentiality, security,
privacy, and legal obligations permit. If BAAI exercises its reserved authority, the
decision and its rationale should be documented publicly to the extent reasonably
possible.

## Conduct Escalation

All participants must follow the [Code of Conduct](CODE_OF_CONDUCT.md). Conduct
incidents should be reported through the FlagCX WeChat group as described there.
Reports should be handled privately. The owner of the WeChat group has final authority
over Code of Conduct enforcement decisions.

## Amending This Document

Anyone may propose a governance change through a pull request, with an accompanying
issue when the change is substantial.

Ordinary clarifications follow the standard pull-request approval process. Substantive
changes require maintainer consensus or, if consensus cannot be reached, a maintainer
vote under the process above. Changes involving a critical decision or BAAI's reserved
authority also require BAAI's approval.
