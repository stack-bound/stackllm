# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [0.0.2] - 2026-04-08
### Added
- Configurable poll intervals and retry backoff (PollInterval on auth configs, BaseBackoff on provider Config, WithPollInterval on profile Manager) to allow fast test execution without hardcoded sleeps
- Provider management layer: config/ package for user preferences, profile/ package composing auth+config+provider with Login/Logout/Status/ListModels/SetDefault/LoadDefault, examples/login interactive CLI, and updated examples to use profile-first resolution with interactive onboarding fallback

## [0.0.1] - 2026-04-08
### Added
- Initial Build

# Notes
[Deployment] Notes for deployment
[Added] for new features.
[Changed] for changes in existing functionality.
[Deprecated] for once-stable features removed in upcoming releases.
[Removed] for deprecated features removed in this release.
[Fixed] for any bug fixes.
[Security] to invite users to upgrade in case of vulnerabilities.
[YANKED] Note the emphasis, used for Hotfixes
