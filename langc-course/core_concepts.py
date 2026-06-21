from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


def demo_basic_chain():
    """Demonstrate a basic chain using LCEL and Runnable."""
    
    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer in one sentance {question}")
    model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Component with pipe operator
    chain = prompt | model | parser

    # Execute the chain

    result = chain.invoke({"question": "What is langchain?"})
    print(f"Response: {result}")

    return chain


if __name__ == "__main__":
    demo_basic_chain()
