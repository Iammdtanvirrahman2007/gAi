from brain.core import GrowingBrain


def main() -> None:
    brain = GrowingBrain()

    print("gAi v0.1 - Human-guided Growing AI")
    print("Commands: learn, request, show, quit")

    while True:
        command = input("\ngAi> ").strip().lower()

        if command == "learn":
            topic = input("Topic: ")
            content = input("What are you teaching? ")
            lesson = brain.learn(topic, content)
            print(f"Learned: {lesson.topic}")

        elif command == "request":
            capability = input("Capability needed: ")
            reason = input("Why is it needed? ")
            raw = input("Requirements (separate with |): ")
            requirements = [item.strip() for item in raw.split("|") if item.strip()]
            request = brain.request_code(capability, reason, requirements)
            print("\n--- CODE REQUEST ---")
            print(request)
            print("--------------------")
            print("Next step: save this request into code_requests/ and have a human implement it.")

        elif command == "show":
            if not brain.lessons:
                print("No lessons yet.")
            else:
                for lesson in brain.lessons:
                    print(f"- {lesson.topic}: {lesson.content}")

        elif command == "quit":
            print("Goodbye.")
            break

        else:
            print("Unknown command.")


if __name__ == "__main__":
    main()
