# Contributing Guidelines

Thank you for your interest in `ovannotators`! Bug reports, feature suggestions, real-world use cases, and technical discussions are highly valuable to us and are the primary way the community shapes this library.

## Questions, Feedback, and Issue Reports

We welcome and actively encourage feedback, question threads, and bug reports via [GitHub Issues](https://github.com/NVIDIA-Omniverse/ovannotators/issues). 

Before opening a new issue, please search the tracker to see if a similar discussion or report already exists.

* **Bug Reports:** Please include the version/commit hash, your platform/environment details, and clear, minimal steps to reproduce the problem.
* **Feature Requests & Suggestions:** Describe your concrete use case, the pain point you are experiencing, and the behavior you would expect to see.

---

## Code Contribution Policy

`ovannotators` is open-source under the Apache-2.0 license, but the repository operates as a read-only public mirror of our active development repository. 

Because of this technical architecture:
* **We cannot merge direct Pull Requests on this repository.** Any open Pull Requests will be closed.
* **Please do not write code patches before aligning on a design.** If you have a fix or a feature idea, describe it in a [GitHub Issue](https://github.com/NVIDIA-Omniverse/ovannotators/issues) first. 

If a suggested improvement aligns with the project’s architectural roadmap and quality standards, our team will integrate the changes into our development branch, allowing them to flow out to the public repository in the next release sync. This policy protects your time as a developer by ensuring you don’t spend hours writing code that technically cannot be merged.

---

## License headers

New source and test files must include the repository SPDX header before code or module docstrings. For Python files, use:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

Before opening a PR, run the OSRB workflow or check locally that every new source file carries `SPDX-FileCopyrightText` and `SPDX-License-Identifier`.

Files generated from third-party tools or copied from another source need explicit provenance and license review before they are committed.
