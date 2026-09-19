from core.memory import remember, get_memories, log_activity


class AkshuRouter:

    def __init__(self):
        self.name = "AKSHU"
        self.version = "3.0"

    def process(self, command):

        if not command:
            return self._response(
                "I didn't receive a command."
            )

        command = command.strip()
        lower_command = command.lower()

        log_activity("command", command)

        # GREETINGS
        if any(
            word in lower_command
            for word in ["hello", "hi", "hey", "namaste"]
        ):
            return self._response(
                "Hello. AKSHU systems are online."
            )

        # SYSTEM STATUS
        if (
            "status" in lower_command
            or "system status" in lower_command
        ):
            return self._response(
                "AKSHU 3.0 is online. "
                "Memory system operational. "
                "Command router operational."
            )

        # REMEMBER
        if lower_command.startswith("remember "):

            value = command[9:].strip()

            if value:
                remember("facts", value)

                return self._response(
                    f"I'll remember that: {value}"
                )

        # PROJECT
        if lower_command.startswith("my project is "):

            project = command[14:].strip()

            if project:
                remember("projects", project)

                return self._response(
                    f"Project registered: {project}"
                )

        if lower_command.startswith("project is "):

            project = command[11:].strip()

            if project:
                remember("projects", project)

                return self._response(
                    f"Project registered: {project}"
                )

        # MEMORY
        if (
            "what do you remember" in lower_command
            or "show memory" in lower_command
            or "show your memory" in lower_command
        ):
            memory = get_memories()

            return self._response(
                self._format_memory(memory)
            )

        # PROJECT QUERY
        if (
            "my projects" in lower_command
            or "what projects" in lower_command
        ):

            projects = get_memories("projects")

            if not projects:
                return self._response(
                    "No projects are currently stored."
                )

            return self._response(
                "Your projects:\n"
                + "\n".join(
                    f"• {project}"
                    for project in projects
                )
            )

        # HELP
        if (
            lower_command == "help"
            or "what can you do" in lower_command
        ):
            return self._response(
                "I can currently:\n"
                "• Remember information\n"
                "• Store projects\n"
                "• Retrieve memory\n"
                "• Report system status\n"
                "• Process commands\n"
                "• Prepare commands for vision and system tools"
            )

        # FALLBACK
        return self._response(
            "Command received, but I don't have "
            "a capability mapped to it yet."
        )

    def _format_memory(self, memory):

        if not memory:
            return "My memory is empty."

        output = []

        for category, values in memory.items():

            if not values:
                continue

            output.append(f"{category.upper()}:")

            if isinstance(values, list):

                for value in values:
                    output.append(f"  • {value}")

            elif isinstance(values, dict):

                for key, value in values.items():
                    output.append(
                        f"  • {key}: {value}"
                    )

            else:
                output.append(f"  • {values}")

        if not output:
            return "My memory is currently empty."

        return "\n".join(output)

    def _response(self, message):

        return {
            "assistant": self.name,
            "version": self.version,
            "message": message
        }

