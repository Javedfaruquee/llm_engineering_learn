import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


def main() -> None:
	load_dotenv(override=False)
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		print(
			"Error: OPENAI_API_KEY is not available to this Python process. "
			"Restart the terminal or IDE after setting the Windows environment variable."
		)
		return

	client = OpenAI(api_key=api_key)

	try:
		response = client.chat.completions.create(
			model="gpt-5-nano",
			messages=[
				{"role": "system", "content": "You are a helpful assistant."},
				{"role": "user", "content": "Explain what an API is in one sentence."},
			],
		)

		if not response.choices or not response.choices[0].message.content:
			print("Error: The API returned an empty response.")
			return

		answer = response.choices[0].message.content.strip()
		print("\n" + "=" * 60)
		print("OpenAI Response")
		print("=" * 60)
		print(answer)
		print("=" * 60)
	except OpenAIError as error:
		print(f"OpenAI API error: {error}")
	except Exception as error:
		print(f"Unexpected error: {error}")


if __name__ == "__main__":
	main()
