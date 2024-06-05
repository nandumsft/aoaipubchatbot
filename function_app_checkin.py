import logging
import azure.functions as func
import azure.functions as func
import logging
from datetime import datetime
import argparse
import base64
import glob
import html
import io
import os
import re
import time
import sys
import pandas as pd 
import ast as ast 

import openai
# from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.storage.blob import BlobServiceClient
from pypdf import PdfReader, PdfWriter
# from tenacity import retry, stop_after_attempt, wait_random_exponential
# import csv
# import requests
# from bs4 import BeautifulSoup
import requests 

MAX_SECTION_LENGTH = 500
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100
filescontainer = "conversation-history"
questionscontainer = "questions"

indexname = ""
files_storage_key = ""
# os.environ["FILES_STORAGE_KEY"] 
# content_storage_key = os.environ["CONTENT_STORAGE_KEY"]
search_key = ""
aoai_key = ""
files_storage_account = "covinodvstochatcc01"
content_storage_account = ""
cogsearch = ""
openaiservice = ""

# def blob_name_from_file_page(filename, page = 0):
#     if os.path.splitext(filename)[1].lower() == ".pdf":
#         return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
#     else:
#         return os.path.basename(filename)

# def upload_blobs(filename):
#     blob_service = BlobServiceClient(account_url=f"https://{storage_account}.blob.core.windows.net", credential=storage_key)

#     # print(filename,file=sys.stderr)
#     blob_container = blob_service.get_container_client(filescontainer)   

#     blob_client = blob_service.get_blob_client(container="pdffiles", blob=filename)
#     stream = io.BytesIO()
#     num_bytes = blob_client.download_blob().readinto(stream)
#     # print(num_bytes, file= sys.stderr)
#     # if file is PDF split into pages and upload each page as a separate blob
#     if (1):
#         reader = PdfReader(stream)
#         # print("Got the pages ",file=sys.stderr)
#         pages = reader.pages
#         for i in range(len(pages)):
#             blob_name = blob_name_from_file_page(filename, i)
#             # if args.verbose: print(f"\tUploading blob for page {i} -> {blob_name}")
#             f = io.BytesIO()
#             writer = PdfWriter()
#             writer.add_page(pages[i])
#             writer.write(f)
#             f.seek(0)
#             # print("Writing file",file=sys.stderr)
#             # with open(blob_name,"wb") as fileout:
#             #     fileout.write(f.getbuffer())
#             # print("Uploading file",file=sys.stderr)
#             blob_container.upload_blob(blob_name, f, overwrite=True)
#     else:
#         blob_name = blob_name_from_file_page(filename)
#         with open(filename,"rb") as data:
#             blob_container.upload_blob(blob_name, data, overwrite=True)

def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

def get_document_text(filename, suffix, container):
    logging.info(f'My app setting value:{files_storage_key}')
    blob_service = BlobServiceClient(account_url=f"https://{files_storage_account}.blob.core.windows.net", credential=files_storage_key)
    blob_container = blob_service.get_container_client(container)
    page_map = []
    offset = 0
    logging.info(f'My app setting value:{files_storage_key}')
    # print(num_bytes, file= sys.stderr)
    logging.info("Extracting text from the file %s ",filename)
    if 'pdf' in filename:
        blob_client = blob_service.get_blob_client(container=container, blob=f"{suffix}/"+filename)

        # print("Trying to get the blob",filename, file=sys.stderr)
        
        stream = io.BytesIO()
        num_bytes = blob_client.download_blob().readinto(stream)
        logging.info("Extract the text from the PDF file")
        reader = PdfReader(stream)
        pages = reader.pages
        page_map = []
        offset = 0
        full_text = ""
        for page_num, p in enumerate(pages):
            
            page_text = p.extract_text()
            full_text = full_text + " "+page_text
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)
        logging.info("The pdf document contains for following text %s",full_text)
        return full_text
    else:
        blob_client = blob_service.get_blob_client(container=container, blob=f"{suffix}/"+filename)

        logging.info("A text file is being used - a job description probably")
        blob_text = blob_client.download_blob(max_concurrency = 1, encoding = 'UTF-8').readall()
        logging.info("The job description is %s ",blob_text)
        return blob_text

def split_text(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
    # if args.verbose: print(f"Splitting '{filename}' into sections")

    def find_page(offset):
        l = len(page_map)
        for i in range(l - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return l - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
            # if args.verbose: print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP
        
    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))


def filename_to_id(filename):
    filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
    filename_hash = base64.b16encode(filename.encode('utf-8')).decode('ascii')
    return f"file-{filename_ascii}-{filename_hash}"

def create_sections(filename, file_content, use_vectors,fileurl):
    file_id = filename_to_id(filename)

    print(file_content)
    publish_time = file_content['PublishDateTime']
    try:
        published_datetime = datetime.strptime(publish_time,"%d/%m/%Y %H:%M:%S AM")
    except:
        published_datetime=datetime.now()
    print("inside create sections",published_datetime)
    sections = []
    file_content
    section = {
            "id": file_content['ID'],
            "content": file_content['Content'],
            "title": file_content['Title'],
            "context": file_content['Context'],
            "type":file_content['Type'],
            "owner": file_content['Owner'],
            "category_service": file_content['Category.Service'],
            "category_category": file_content['Category.Category'],
            "category_subcategory":file_content['Category.Subcategory'],
            "publishedtime":published_datetime,
            "status":file_content['Status'],
            "filepath":fileurl
        }
    
    if use_vectors:
            print("Gettng embeddings",file=sys.stderr)
            section["embedding"] = compute_embedding(file_content['Content'])
            section["titleembedding"] = compute_embedding(file_content['Title'])
            section['contextembedding'] = compute_embedding(file_content['Context'])
            
    sections.append(section)
    print("Got the sections",file=sys.stderr)
    return sections


def generate_questions(content):
    from openai import AzureOpenAI

    import os
    # import openai
    # openai.api_type = "azure"
    # openai.api_version = "2024-02-01" 
    # openai.api_base = f"https://{openaiservice}.openai.azure.com/"  # Your Azure OpenAI resource's endpoint value.
    # openai.api_key = aoai_key
    client = AzureOpenAI(azure_endpoint = f"https://{openaiservice}.openai.azure.com/", api_key = aoai_key, api_version='2024-03-01-preview')
    response = client.chat.completions.create(
        model="gpt35turbo1106", # model = "deployment_name".
        response_format={ "type": "json_object" },
        max_tokens = 2000,
        messages=[
            {"role": "system", "content": "Given a conversation find the questions asked and referenced URL associated with the conversation that are related to City of Vancouver, Parks, general public services. Only extract the key enquiry or concern or issues that are associated with public services of vancouver. Find the exact questions asked in the conversation and generate formal variant of the questions.  If required add additional information on the topic associated with the conversation, any names, keywords. For example if the conversation contains https://vancouver.ca/parks-recreation-culture/bloedel-conservatory.aspx, the topic should regarding Bloedel Conservatory.  Generate the answer only with the given below conversation.  Respond with a json with list of asked questions, formal questions and the corresponding url. \
              An example of the final response would be this form:  [{'Asked Questions': Is the train working today?, 'Formal Question': Is the tourist train in Stanley Park operational today?, 'URL': https://vancouver.ca/parks-recreation-culture/temporary-bike-lane-on-stanley-park-drive.aspx}] or if multiple questions are generated, then format the output in this form: [{'Asked Question': 'Can I pay my parking ticket online?', 'Formal Question': 'Is there an option to pay parking tickets online?', 'URL': 'https://vancouver.ca/streets-transportation/pay-your-ticket.aspx'},  {'Asked Question': 'How can I dispute a parking ticket?', 'Formal Question': 'Is there a process to dispute a parking ticket?', 'URL': 'https://vancouver.ca/streets-transportation/parking-ticket-adjudication.aspx'}]"},
            {"role": "user", "content": content}
        ]
    )


    #print(response)
    logging.info("The qualifications are %s ", response.choices[0].message.content)
    return response.choices[0].message.content




def compute_embedding(text):
    openai.api_type="azure"
    #Add the openai key and service
    openai.api_key=aoai_key
    openai.api_base = f"https://{openaiservice}.openai.azure.com/"
    openai.api_version = "2022-12-01"
    return openai.Embedding.create(engine="embedding", input=text)["data"][0]["embedding"]



def index_sections(indexname, sections):
    """ Index the sections """
    search_creds = AzureKeyCredential(search_key)

    print("Inside indexing ",indexname,file=sys.stderr)
    # if args.verbose: print(f"Indexing sections from '{filename}' into search index '{args.index}'")
    search_client = SearchClient(endpoint=f"https://{cogsearch}.search.windows.net",
                                    index_name=indexname,
                                    credential=search_creds)
    print("Inside indexing ",indexname,file=sys.stderr)
    
    i = 0
    batch = []
    print("Length of sections ",len(sections),file=sys.stderr)
    for s in sections:
        batch.append(s)
        i += 1
        if i % 1000 == 0:
            print("Uploading 1",file=sys.stderr)
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            if 1: print(f"\tIndexed {len(results)} sections, {succeeded} succeeded",file=sys.stderr)
            batch = []

    if len(batch) > 0:
        print("Uploading 2",file=sys.stderr)

        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        if 1: print(f"\tIndexed {len(results)} sections, {succeeded} succeeded",file=sys.stderr)

def delete_from_index(filename, indexname):
    search_creds = AzureKeyCredential(search_key)
    search_client = SearchClient(endpoint=f"https://{cogsearch}.search.windows.net",
                                    index_name=indexname,
                                    credential=search_creds)
    results = search_client.search("*", select="filename",filter=f"filename eq '{filename}")
    file = [[json.loads[results["filename"]]]]
    if(len(file)):
        return True 
    else:
        return False
def remove_from_index(indexname,filename):
    search_creds = AzureKeyCredential(search_key)
    # if args.verbose: print(f"Removing sections from '{filename or '<all>'}' from search index '{args.index}'")
    search_client = SearchClient(endpoint=f"https://{cogsearch}.search.windows.net/",
                                    index_name=indexname,
                                    credential=search_creds)
    while True:
        filter = None if filename == None else f"sourcefile eq '{filename}'"
        r = search_client.search("", filter=filter, top=1000, include_total_count=True)
        if r.get_count() == 0:
            break
        r = search_client.delete_documents(documents=[{ "id": d["id"] } for d in r])
        # It can take a few seconds for search results to reflect changes, so wait a bit
        time.sleep(2)

import json 
# app = func.FunctionApp()

# @app.event_grid_trigger(arg_name="azeventgrid")
# def BlobEventGridTrigger(azeventgrid: func.EventGridEvent):
#     result = json.dumps({
#         'id': azeventgrid.id,
#         'data': azeventgrid.get_json(),
#         'topic': azeventgrid.topic,
#         'subject': azeventgrid.subject,
#         'event_type': azeventgrid.event_type,
#     })
import csv
def run():
    blob_service = BlobServiceClient(account_url=f"https://{files_storage_account}.blob.core.windows.net", credential=files_storage_key)
    blob_container = blob_service.get_container_client(filescontainer)
    qns_container = blob_service.get_container_client(questionscontainer)
    blob_list = blob_container.list_blobs(name_starts_with='transcripts/2024')
    list_of_files = []  
    count = 1
    df = pd.DataFrame(columns=['Asked Question', 'Formal Question','URL','conversation'])
    df.to_csv("questions.csv" , sep=',', encoding='utf-8', doublequote=False, index=False,header=True)
    for f in blob_list:
        if("2024" in f.name):
            # print(f.name) 

            count +=1
            filename = f.name
            blob_client = blob_service.get_blob_client(container = filescontainer, blob = filename)

            filepath=(blob_client.url)
            # print(filepath)
            # csv_file = pd.read_csv(filepath)
            # csv_file_text = csv_file['message'].to_list()
            # print(csv_file_text)
            blob_text = blob_client.download_blob().readall()
            # json_format=ast.literal_eval(generate_questions(blob_text.decode('utf-8')))
            json_format = json.loads(generate_questions(blob_text.decode('utf-8')))
            final_dict=[]
            if len(json_format) == 1:
                        print('a list element')                     
                        for k,v in json_format.items():
                            for item in v:
                                final_dict.append(item)
            else:
                        print('single elements')
                        final_dict.append(json_format)
            ind = 0
            print(final_dict)
            for item in final_dict:
                if 'URL' in item:
                    if (item['URL'] ):
                        ind+=1
                        item['conversation'] = filename
                with open('questions.csv','a') as f:
                    w = csv.DictWriter(f,item.keys())
                    w.writerow(item)
    with open('questions.csv', mode ='rb') as data:
        qnblob_client = qns_container.upload_blob(name = "questions.csv", data = data, overwrite = True)
        print(qnblob_client.url) 
run()