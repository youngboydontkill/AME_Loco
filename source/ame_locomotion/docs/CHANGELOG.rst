Changelog
---------

0.1.0 (2025-09-08)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab
* Added terrain-curriculum diagnostics for AME/S54 depth training and fixed the S54 bootstrap path so the policy can actually learn forward motion instead of only standing in place.
* Narrowed S54 illegal-contact termination to the true base links and disabled destabilizing depth-task reset pushes/joint velocity noise during bootstrap.
