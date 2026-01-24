---
description: Task Management Workflow using .tickets and .complete
---

Follow this structured workflow for all new development tasks:

1. **Task Initiation**
   Create a markdown file in `.tickets/` for each task (e.g., `.tickets/feature-xyz.md`).

2. **User Summary**
   Ask the user to provide an initial summary of the work to be done if not already provided.

3. **Clarification & Review**
   Ask for clarification and provide an initial review of the requirements in the ticket file.

4. **AI Summary**
   Write a comprehensive summary of the work, including scope, requirements, and approach, directly into the ticket file.

5. **User Review**
   Wait for the user to review and approve the AI summary.

6. **Checklist Creation**
   Add a detailed checklist of work items to the ticket file.

7. **Implementation**
   Once instructed to proceed, update the checklist with progress as work is completed.
   - **Important**: If the work requires changes or updates to the project design, always keep the design docs in `docs/design/overview.md` up to date.

8. **Completion**
   When the ticket is complete, move the ticket task file from `.tickets/` to `.complete/`.

---

### Ticket Template
```markdown
# [Task Title]

## Overview
[User summary of the work]

## AI Clarification Questions
- [Question 1]
- [Question 2]

## AI Review
[Initial review of requirements]

## AI Summary
**Scope**: [Description]
**Requirements**: 
- [Req 1]
**Approach**:
1. [Step 1]

## Project Plan / Checklist
- [ ] [Phase/Task 1]
- [ ] [Phase/Task 2]
```
