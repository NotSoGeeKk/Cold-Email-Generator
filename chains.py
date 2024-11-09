import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Replace 'YOUR_API_KEY_HERE' with your actual API key
groq_api_key = "USE_YOUR_API_KEY_HERE"

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            You are Samarth Solanki studying at ljiet in ahmedabad currently in final year of BE(CSE).Samarth is excellent student
            and highly skilled in data analytics with previous experience in previous internships at ignitus and compwallah as 
            well as doing high end projects such as creating an resume analyzer and creating a chat pdf and also creating a cold 
            email generator.
            Your job is to write a cold email to the client regarding the job mentioned above and craft it in such a way that it 
            doeesnt seem ai generated and fullfill all the needs of the client.
            also mention my email:samarthsolanki1731@gmail.com and my linked in portfolio as:https://www.linkedin.com/in/samarthsolanki/
            in my contact details.Do not provide a preamble.
            
            ##EMAIL(NO PREAMBLE)
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job)})
        return res.content

if __name__ == "__main__":
    print(groq_api_key)
