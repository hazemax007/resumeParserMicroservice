from flask import Flask, request, jsonify
import docx2txt
import PyPDF2
import logging
import parser
import spacy
from spacy.matcher import Matcher
import re
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import os
from nltk.tokenize import word_tokenize
import re


nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

app = Flask(__name__)


def convert_pdf_to_txt(pdf_file):

    try:


        raw_text = parser.from_file(pdf_file,service='text')['content']
        

        full_string = re.sub(r'\n+','\n',raw_text)
        full_string = full_string.replace("\r", "\n")
        full_string = full_string.replace("\t", " ")

        # Remove awkward LaTeX bullet characters
        
        full_string = re.sub(r"\uf0b7", " ", full_string)
        full_string = re.sub(r"\(cid:\d{0,2}\)", " ", full_string)
        full_string = re.sub(r'• ', " ", full_string)

        # Split text blob into individual lines
        resume_lines = full_string.splitlines(True)

        # Remove empty strings and whitespaces
        resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if line.strip()]

        return resume_lines, raw_text

    except Exception as e:
        logging.error('Error in pdf file:: ' + str(e))
        return [], " "

        
def convert_docx_to_txt(docx_file):

 
    try:
        
        text = parser.from_file(docx_file,service='text')['content']        
        clean_text = re.sub(r'\n+','\n',text)
        clean_text = clean_text.replace("\r", "\n").replace("\t", " ")  # Normalize text blob
        resume_lines = clean_text.splitlines()  # Split text blob into individual lines
        resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if line.strip()]  # Remove empty strings and whitespaces
        return resume_lines ,text
    except Exception as e:        
        logging.error('Error in docx file:: ' + str(e))
        return [], " "
    

STOPWORDS = set(stopwords.words('english')+['``',"''"])
def clean_text(resume_text):
        resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
        resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
        resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
        resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
        resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
        resume_text = re.sub(r'[^\x00-\x7f]',r' ', resume_text) 
        resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace
        resume_text = resume_text.lower()  # convert to lowercase
        resume_text_tokens = word_tokenize(resume_text)  # tokenize
        filtered_text = [w for w in resume_text_tokens if not w in STOPWORDS]  # remove stopwords
        return ' '.join(filtered_text)


def extract_email(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)


def extract_name(text):
   nlp_text = nlp(text)
  
   # First name and Last name are always Proper Nouns
   pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
  
   matcher.add('NAME', [pattern], on_match = None)
  
   matches = matcher(nlp_text)
  
   for match_id, start, end in matches:
       span = nlp_text[start:end]
       return span.text
   

def extract_mobile_number(text):
     phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
     if phone:
         number = ''.join(phone[0])
         if len(number) > 10:
             return '+' + number
         else:
             return number
         

data= pd.read_csv("newskill2.csv") 
SKILLS_DB = list(data.columns.values)
def extract_skills(input_text):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = nltk.tokenize.word_tokenize(input_text)

        # remove the stop words
        filtered_tokens = [w for w in word_tokens if w not in stop_words]

        # remove the punctuation
        filtered_tokens = [w for w in word_tokens if w.isalpha()]

        # generate bigrams and trigrams (such as artificial intelligence)
        bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

        # we create a set to keep the results in.
        found_skills = set()

        # we search for each token in our skills database
        for token in filtered_tokens:
            if token.lower() in SKILLS_DB:
                found_skills.add(token)

        # we search for each bigram and trigram in our skills database
        for ngram in bigrams_trigrams:
            if ngram.lower() in SKILLS_DB:
                found_skills.add(ngram)
        set_text = ', '.join(found_skills)
        return set_text


def extract_resume_category(resume_text):
  
    # Parse the resume using spaCy
    doc = nlp(resume_text)

    # Initialize variables
    headings = []
    keywords = []

    # Extract headings and keywords
    for token in doc:
        if token.is_stop:
            continue
        if token.is_title and token.is_alpha:
            headings.append(token.text)
        if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
            keywords.append(token.lemma_.lower())

    # Define categories and their corresponding keywords
    category_keywords = {
        'Full-Stack Engineer': ['EngineeringOracle', 'Invoices', 'access', 'workflow', 'Engineeringanalytical', 'inventory', 'Computer Science', 'JSP', 'Javaanalysis', 'Oracle', 'MIS', 'visual', 'XML', 'user stories', 'Information Technologydatabase', 'Technical Skills', 'ProgrammingOracle', 'Engineering', 'CSS', 'billing', 'Sales', 'Android', 'payments', 'Java', 'English', 'Engineeringjava', 'ANDROID', 'Scripting', 'MySql', 'PROGRAMMING', 'PHP', 'SQL', 'MySQL', 'Access', 'WindowsCSS', 'js', 'Linux', 'Billing', 'C', 'process', 'purchasing', 'communication', 'video', 'Excel', 'Mobile', 'testing', 'engineering', 'Life Cycle', 'JSON', 'Information Technology', 'Banking', 'WindowsJavascript', 'open Source', 'Sql', 'Shell', 'System', 'Sublime', 'Operating Systems', 'JavaJs', 'website', 'JS', 'HTML', 'Json', 'CommunicationJAVA', 'Mysql', 'Programming', 'Architecture', 'Facebook', 'ordering', 'documentation', 'Jsp', 'Technical', 'Migration', 'Web services', 'Payments', 'Testing', 'oracle', 'routing', 'ISOJsp', 'Swift', 'DATABASE', 'Windows', 'Website', 'Electronics', 'EngineeringCSS', 'jsp', 'analytical skills', 'analysis', 'inventory management', 'TECHNICAL', 'sales', 'JavaOracle', 'design', 'TECHNICAL SKILLS', 'architecture', 'Visual', 'security', 'OS', 'invoices', 'automation', 'invoicing', 'social media', 'technical', 'JavaScript', 'system', 'Javascript', 'java'],
        'DevOps Engineer': ['Reporting', 'AI', 'access', 'Software Engineering', 'Documentation', 'workflow', 'proposal', 'IBM', 'policies', 'Process', 'Computer Science', 'SAP', 'JSP', 'Transport', 'Information security', 'electronics', 'Redhat', 'Tableau', 'compliance', 'Oracle', 'XML', 'MATLAB', 'EngineeringAWS', 'Application Support', 'Security', 'Pattern', 'Audit', 'linux', 'Project Delivery', 'Controls', 'Project Management', 'Accounting', 'ERP', 'Technical Skills', 'Analyze', 'queries', 'Engineering', 'JAVA', 'CSS', 'AWS', 'operations', 'Strategy', 'billing', 'Budgeting', 'Sales', 'Android', 'Web Services', 'Resource Management', 'Machine learning', 'improvement', 'training', 'GitHub', 'Java', 'Information System', 'English', 'recruit', 'Automation', 'Scripting', 'schedules', 'MySql', 'PHP', 'Economics', 'SQL', 'TFS', 'information technology', 'Time Management', 'Presentation', 'mysql', 'Writing', 'Linux', 'WindowsReporting', 'Ansible', 'Billing', 'C', 'process', 'regulatory', 'Risk Management', 'communication', 'Mobile', 'Matlab', 'Reports', 'testing', 'windows', 'Consulting', 'Life Cycle', 'JSON', 'iOS', 'Project management', 'quality management', 'debugging', 'specifications', 'Information Technology', 'International', 'Banking', 'Design', 'MVP', 'Agile', 'cryptography', 'performance metrics', 'payroll', 'Cloud', 'Shell', 'JIRASelenium', 'Conversion', 'reporting', 'System', 'Operations', 'Operating Systems', 'website', 'Data Management', 'JS', 'reports', 'Cisco', 'Troubleshooting', 'UI', 'SQL Server', 'HTML', 'ASP', 'Programming', 'Mysql', 'Architecture', 'Networking', 'documentation', 'metrics', 'HP ALM', 'Health', 'forecasting', 'Technical', 'engagement', 'banking', 'research', 'Servers', 'Testing', 'Plan', 'marketing', 'JIRA', 'analytics', 'database', 'scripting', 'regulatory compliance', 'Scrum', 'key performance indicators', 'Human Resource', 'Quality assurance', 'Administration', 'Analysis', 'API', 'Root Cause', 'Robot', 'SharePoint', 'technical Knowledge', 'Windows', 'Engineeringdatabase', 'audit', 'Electronics', 'plan', 'c', 'Invoicing', 'Quality Assurance', 'Metrics', 'application support', 'analysis', 'shell', 'Information SecuritySelenium', 'TECHNICAL', 'sales', 'Ubuntu', 'servers', 'EngineeringJAVA', 'design', 'Procurement', 'architecture', 'TECHNICAL SKILLS', 'distribution', 'CRM', 'Visual', 'Scheduling', 'security', 'Program Management', 'automation', 'Python', 'Finance', 'technical', 'JavaScript', 'Process Improvements', 'system', 'EngineeringReporting', 'Javascript', 'KPI', 'Account Management', 'Database', 'vendors', 'Business Continuity'],
        'Web Designing': ['Photography', 'sublime', 'Website', 'bankingInformation Technologyanalysis', 'Corel Draw', 'international', 'technical', 'SDLC', 'Security', 'Illustrator', 'Advertising', 'tax', 'Operating Systems', 'System', 'Life cycle', 'bankingphp', 'mobile', 'Agile', 'vendors', 'specifications', 'Technical', 'JS', 'ASP', 'Information technology', 'health', 'process', 'website', 'Technical Skills', 'design', 'Health', 'documentation', 'system', 'Click', 'reconciliation', 'Windows', 'CSS', 'UI', 'Adobe', 'PHP', 'Database', 'C', 'Presentation', 'photoshop', 'access', 'HTML', 'sales', 'JAVA', 'SQLPhotography', 'Engineering', 'Sublime', 'Visual', 'expenses', 'Real Estate', 'Design', 'JavaScript', 'MySQL', 'javascript', 'SYSTEM', 'Photoshop', 'Javascript', 'sublimePhotography'],
        'Data Scientist': ['Selenium', 'Jupyter', 'machine learning', 'Reporting', 'pattern', 'AI', 'researchELECTRICAL', 'access', 'six', 'IBM', 'Hadoop', 'Process', 'Flower', 'Computer Science', 'SAP', 'expenses', 'Tableau', 'Statistics', 'EngineeringAlgorithms', 'NLTK', 'Oracle', 'MATLAB', 'Predictive analytics', 'MYSQL', 'ggplot', 'Pycharm', 'R', 'Keras', 'computer science', 'Flask', 'Governance', 'Microsoft Word', 'ERP', 'Accounting', 'Github', 'EngineeringDatabasePython', 'Technical Skills', 'queries', 'CSS', 'analytical', 'AWS', 'predictive analytics', 'Docker', 'Machine learning', 'training', 'mobile', 'Java', 'Engineeringanalytics', 'English', 'marketing strategy', 'Communication', 'Queries', 'numpy', 'PySpark', 'SWIFT', 'SQL', 'Mining', 'MySQL', 'Electrical', 'Modeling', 'Powerpoint', 'Word', 'Linux', 'financial reports', 'C', 'process', 'content', 'strategy', 'communication', 'Excel', 'python', 'Matlab', 'Segmentation', 'Reports', 'Research', 'Information Technology', 'queriesCSS', 'analyze', 'cloud', 'Sql', 'Design', 'ANALYTICS', 'GOVERNANCE', 'interactive', 'Business Intelligence', 'Cloud', 'reporting', 'System', 'Microsoft Excel', 'Sublime', 'Operating Systems', 'Analytics', 'spacy', 'reports', 'Matplotlib', 'matplotlib', 'Healthcare', 'SQL Server', 'HTML', 'ETL', 'pandas', 'Programming', 'Mysql', 'Retail', 'Twitter', 'documentation', 'Hive', 'forecasting', 'ExcelPython', 'Technical', 'engagement', 'Training', 'research', 'PR', 'Testing', 'Big Data', 'Spark', 'marketing', 'responses', 'analytics', 'database', 'tableau', 'Machine Learning', 'Analysis', 'Numpy', 'API', 'Windows', 'coding', 'mining', 'Telecom', 'Product development', 'Electronics', 'Forecasting', 'TRAINING', 'consulting', 'Hbase', 'analysis', 'segmentation', 'TECHNICAL', 'Forecasts', 'Teaching', 'Ubuntu', 'Microsoft Powerpoint', 'design', 'Investigations', 'servers', 'TECHNICAL SKILLS', 'CRM', 'KPIs', 'distribution', 'development activities', 'scipy', 'Visual', 'programming', 'security', 'Seaborn', 'responsesforecasting', 'invoices', 'automation', 'Python', 'Finance', 'technical', 'JavaScript', 'system', 'KPI', 'Tensorflow', 'TensorFlow', 'algorithms', 'Pandas', 'Database', 'RETAIL'],
        'HR': ['engagement', 'analysis', 'compliance', 'excel', 'filing', 'computer science', 'TECHNICAL', 'Programming', 'reports', 'Recruitment', 'recruitment', 'administration', 'tax', 'Word', 'Payroll', 'payroll', 'Interactive', 'Pivot', 'budgetEconomicsEconomicsengineeringscrum', 'TECHNICAL SKILLS', 'Employee Relations', 'Ecommerce', 'accounting', 'communication', 'CTECHNICAL SKILLS', 'Policies', 'DATABASE', 'process', 'Hospitality', 'Excel', 'Compliance', 'CTraining', 'Training', 'Employee engagement', 'training', 'Sales', 'Windows', 'plan', 'schedule', 'PsychologyAdministration', 'engineering', 'C', 'Hotel', 'Human resourceAdministration', 'queries', 'EngineeringJava', 'Engineering', 'International', 'Finance', 'MATLAB'],
        'Mechanical Engineer': ['product development', 'modeling', 'Modeling', 'English', 'cement', 'Auto CAD', 'ERP', 'proposalmechanical engineering', 'Microsoft officedesign', 'design', 'customer requirements', 'Fabrication', 'supply chain Management', 'Brand', 'ENGINEERING', 'schedule', 'plan', 'Continuous improvement', 'Procurement', 'procurement', 'Travel', 'Engineering', 'construction', 'Contract Management', 'Autocad', 'Microsoft Office', 'Programming', 'contracts', 'invoices', 'customer service', 'specifications', 'Supply Chain', 'engineeringMechanical Engineering', 'Assembly', 'Excel', 'requests', 'cost effective', 'Training', 'documentation', 'Sales', 'training', 'system', 'cementfield sales', 'Instrumentation', 'Prototype', 'Cement', 'sales', 'budget', 'marketing', 'Mechanical engineering', 'field sales', 'analysis', 'Vendors', 'EngineeringResearch', 'analyze', 'System', 'vendors', 'Automation', 'Technical', 'process', 'audit', 'Research', 'cad', 'Mechanical Engineering', 'solidworks', 'engineering', 'internal audit', 'improvement', 'standard operating procedures', 'CAD', 'FITNESS', 'technical', 'reports', 'MECHANICAL ENGINEERING', 'Proposal', 'invoicing', 'inventory', 'Auto cad', 'R', 'supply chain', 'ISO', 'fabrication', 'purchasing', 'Design', 'database', 'electrical', 'negotiation'],
        'Business Analyst': ['JIRA', 'testing', 'Visio', 'presentations', 'Operating Systems', 'Modeling', 'English', 'research', 'Cloud', 'Supervising', 'inventory management', 'Email', 'Test Case', 'Plan', 'ERP', 'Asset Management', 'test cases', 'gap analysis', 'reporting', 'design', 'PowerPoint', 'plan', 'email', 'matrix', 'Test cases', 'Inventory management', 'Exceltesting', 'Accounting', 'queries', 'HTML', 'prototype', 'JAVA', 'Engineering', 'Payments', 'JavaScript', 'Risk Assessment', 'excel', 'Microsoft Office', 'Audit', 'payroll', 'Project Plan', 'invoices', 'Pharmaceutical', 'business process', 'specifications', 'Merchant', 'operations management', 'Reporting', 'Internal Audit', 'Excel', 'Training', 'UNIX', 'regulations', 'documentation', 'training', 'system', 'Sales', 'Sales Management', 'Database', 'ExcelMIS', 'Information Management', 'AutoCAD', 'sales', 'financing', 'transactions', 'Electronics', 'SQL', 'analysis', 'operationsrisk management', 'Scrum', 'Servers', 'SDLC', 'OS', 'analyze', 'Word', 'proposalpresentation', 'System', 'Supply chain', 'Agile', 'Java', 'vendors', 'Technical', 'Automation', 'accounting', 'communication', 'process', 'audit', 'SQLanalysis', 'life cycle', 'Research', 'Technical Skills', 'Computer Science', 'Testing', 'PHP', 'internal audit', 'Specifications', 'Process', 'improvement', 'German', 'technical issues', 'Android', 'project delivery', 'Prototyping', 'process improvement', 'banking', 'Metrics', 'Quality assurance', 'Data Management', 'technical', 'workflow', 'recruitment', 'reports', 'Proposal', 'inventory', 'Payroll', 'Inventory', 'MIS', 'Communication', 'Vendor Management', 'Analyze', 'Chemicals', 'Paymentsanalysis', 'Logistics', 'Documentation', 'Networking', 'Writing', 'test plans', 'Windows', 'Gap Analysis', 'UI', 'MS Excel', 'Drafting', 'end user', 'Oracle', 'IOS', 'CRM', 'Analysis', 'Design', 'database', 'health', 'Excelanalysis', 'writing', 'automation', 'Mobile', 'Matrix'],
        'DotNet Developer': ['testing', 'TECHNICAL', 'Scripting', 'Security', 'architecture', 'Operating Systems', 'SQL SERVER', 'ASPHTML', 'Access', 'CTECHNICAL', 'hospital', 'Coaching', 'C', 'Travel', 'access', 'HTML', 'JavaScript', 'Mortgage', 'ASP', 'Transport', 'programming', 'Programming', 'logistics', 'JSON', 'controls', 'Reporting', 'JS', 'Excel', 'website', 'documentation', 'agile', 'training', 'system', 'CSS', 'Database', 'Presentation', 'coding', 'Hospitalscheduling', 'sql', 'SQL', 'operations', 'Cpresentation', 'asp', 'Coding', 'Word', 'sql server', 'System', 'TECHNICAL SKILLS', 'SQL ServerERPcontent', 'Technical', 'accounting', 'sports', 'process', 'Technical Skills', 'Computer Science', 'CMS', 'Testing', 'coaching', 'Visual', 'RestWebsite', 'css', 'technical', 'reports', 'administration', 'API', 'Legal', 'cms', 'recruit', 'PDF', 'Windows', 'UI', 'schedules', 'interactive', 'html', 'CRM', 'c', 'relationship management', 'Design', 'database', 'automation', 'SQL Server'],
        'Automation Testing': ['JIRA', 'testing', 'TECHNICAL', 'acquisition', 'SQLAdministration', 'Microsoft SQL', 'Scripting', 'Time Management', 'SQL SERVERAdministration', 'Life CycleAdministration', 'Email', 'commissioning', 'Hardware', 'Test Case', 'ERP', 'test cases', 'reporting', 'design', 'hospital', 'Machine Learning', 'Jsp', 'test casesbottle', 'plan', 'schedule', 'status reports', 'Lifecycle', 'consulting services', 'email', 'matrix', 'Test cases', 'C', 'consulting', 'JAVA', 'HTML', 'Engineering', 'electricalprogramming', 'JavaScript', 'Banking', 'Installation', 'Linux', 'programming', 'Programming', 'logistics', 'test case', 'MONEY', 'Protocols', 'Retail', 'Merchant', 'Reporting', 'Updates', 'Life CycleSelenium', 'Healthcare', 'DATABASE', 'Excel', 'Selenium', 'Training', 'MatrixSELENIUM', 'Telecom', 'Health', 'regulations', 'Sales', 'system', 'strategy', 'Instrumentation', 'Database', 'usability', 'AutoCAD', 'sales', 'Real Estate', 'Test Cases', 'Electronics', 'SQL', 'Scrum', 'FDA', 'TRAINING', 'SDLC', 'OS', 'analyze', 'SQL server', 'Information Technology', 'System', 'Agile', 'oracle', 'Java', 'TECHNICAL SKILLS', 'Automation', 'Technical', 'process', 'Quality Control', 'life cycle', 'hardware', 'Research', 'Technical Skills', 'Computer Science', 'Life CycleWebsite', 'Testing', 'engineering', 'Shell', 'scripting', 'Android', 'installation', 'Python', 'Metrics', 'Money', 'technical', 'Jira', 'Construction', 'reports', 'VB Script', 'recruitment', 'invoicing', 'inventory', 'API', 'Debugging', 'Inventory', 'Electrical', 'XML', 'MIS', 'security', 'ElectronicsSelenium', 'Robot', 'Inventory Management', 'Communication', 'Invoicing', 'logging', 'Negotiation', 'Logistics', 'Reports', 'Windows', 'Oracle', 'Analysis', 'Design', 'database', 'automation', 'writing', 'MySQL', 'Mobile', 'AUTOMATION', 'SQL Server', 'Matrix', 'Life Cycle'],
        'Network Security Engineer': ['certification', 'Cisco', 'Security', 'writingPython', 'Operating Systems', 'English', 'VMware', 'CISCO', 'Cloud', 'routing', 'Content', 'inventory management', 'commissioning', 'Hardware', 'Plan', 'Access', 'design', 'customer requirements', 'EMEA', 'Ubuntu', 'Accounting', 'access', 'queries', 'researching', 'Engineering', 'P', 'Installation', 'Administration', 'windowstroubleshooting', 'Linux', 'data center', 'LAN', 'analytical skills', 'Protocols', 'analytical', 'Reporting', 'INFORMATION TECHNOLOGY', 'Policies', 'requests', 'Change management', 'Logging', 'networking', 'Health', 'system', 'DNS', 'DHCP', 'analysis', 'Servers', 'OS', 'analyze', 'Information Technology', 'System', 'Routing', 'vendors', 'Technical', 'accounting', 'communication', 'process', 'writingVMWare', 'hardware', 'WiFi', 'Active Directory', 'ACCESS', 'ITIL', 'servers', 'updates', 'installation', 'windows', 'troubleshooting', 'ElectronicsRouting', 'technical', 'reports', 'administration', 'inventory', 'API', 'Lan', 'MIS', 'security', 'Communication', 'Analyze', 'policies', 'Documentation', 'Networking', 'Windows', 'schedules', 'Oracle', 'Troubleshooting', 'SystemAdministration', 'IOS', 'Finance'],
     
        # Add more categories and their keywords as needed
    }

    # Perform keyword analysis to determine the category
    predicted_category = None
    max_keyword_matches = 0

    for category, category_kw in category_keywords.items():
        keyword_matches = len(set(keywords) & set(category_kw))
        print(keyword_matches)
        if keyword_matches > max_keyword_matches:
            predicted_category = category
            max_keyword_matches = keyword_matches

    return predicted_category




@app.route('/', )
def index():
    return 'Hello, Flask!'


@app.route('/extractor', methods=['POST'])
def extractor():
    file = request.files['resume']
    ext = os.path.splitext(file.filename)[1][1:].lower()

    if ext == "docx":
        temp = docx2txt.process(file)
        text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
        text = ' '.join(text)
    elif ext == "pdf":
        file_path = os.path.join(os.getcwd(), file.filename)
        file.save(file_path)
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(reader.pages)
            text = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
        text = " ".join(text.split('\n'))
        os.remove(file_path)
    else:
        return jsonify({'error': 'Invalid file format'})

    email = extract_email(text)
    text = clean_text(text)
    name = extract_name(text)
    mobile_no = extract_mobile_number(text)
    skills = extract_skills(text)
    category = extract_resume_category(skills)

    result = {
        'name': name,
        'mobile_no': mobile_no,
        'email': email,
        'skills': skills,
        'category': category
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run()
