from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")
    information = """
    Jesus[e] (c. 6 to 4 BC – AD 30 or 33), also known as Jesus Christ,[f] Jesus of Nazareth, and by various other names and titles, was a Jewish preacher and religious leader active in the Roman province of Judaea during the first century.
    [12] He is the central figure of Christianity, the world's largest religion. Most branches of Christianity consider Jesus the incarnation of God the Son and the awaited Messiah or Christ. Accounts of Jesus's life are contained in the gospels, especially the four canonical gospels of the New Testament. Since the Enlightenment, academic research has produced various views on the historical reliability of the gospels and the extent to which they reflect the historical Jesus,
    [13][g][16][17] but virtually all modern scholars of antiquity agree that Jesus existed historically.[h]
"""

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary of the person in 2-3 sentences.
    2. A list of 5 interesting facts about the person.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    #llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.8-flash")
    llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm

    result = chain.invoke({"information": information})
    print(result.content)
if __name__ == "__main__":
    main()
