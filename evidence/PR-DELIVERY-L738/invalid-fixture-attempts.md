# Invalid investigation attempts, not evidence of planner behavior

- Initial test-file edit invocations failed during shell quoting before touching files; they did not run tests.
- First Queue.host_plan planner probe used unregistered repository identities, so it failed during Queue construction before planner invocation. The direct planner control then used the exact `github.com/aien-dev/aien-architecture` spelling and committed L700 fixture head/base. Only planner-default/operator logs are valid planner evidence.
