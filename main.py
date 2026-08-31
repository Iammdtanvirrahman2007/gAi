from brain.core import GrowingBrain


def print_lessons(brain: GrowingBrain) -> None:
    if not brain.lessons:
        print("No lessons yet.")
        return
    print("\n=== MEMORY ===")
    for index, lesson in enumerate(brain.lessons, 1):
        print(f"{index}. {lesson.topic}")
        print(f"   {lesson.content}")
        print(f"   learned: {lesson.learned_at}")


def main() -> None:
    brain = GrowingBrain()

    print("gAi v0.2 - Human-guided Growing AI")
    print("The AI can learn and request code, but it does NOT write implementation code.")
    print("Commands: learn, request, search, show, quit")

    while True:
        command = input("\ngAi> ").strip().lower()

        try:
            if command == "learn":
                topic = input("Topic: ")
                content = input("What are you teaching? ")
                lesson = brain.learn(topic, content)
                print(f"Learned and saved: {lesson.topic}")

            elif command == "request":
                capability = input("Capability needed: ")
                reason = input("Why is it needed? ")
                target_file = input("Required file path (or TBD): ")
                raw = input("Requirements (separate with |): ")
                requirements = [item.strip() for item in raw.split("|")]
                path = brain.request_code(capability, reason, requirements, target_file)
                print("\nCODE REQUEST CREATED")
                print(f"File: {path.relative_to(brain.memory_file.parent.parent)}")
                print("Status: WAITING_FOR_HUMAN_CODE")
                print("The AI will not write this implementation.")

            elif command == "search":
                query = input("Search memory: ").strip()
                results = brain.search_memory(query)
                if not results:
                    print("No matching lessons.")
                else:
                    for lesson in results:
                        print(f"- {lesson.topic}: {lesson.content}")

            elif command == "show":
                print_lessons(brain)

            elif command == "quit":
                print("Goodbye.")
                break

            else:
                print("Unknown command. Use: learn, request, search, show, quit")

        except ValueError as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
