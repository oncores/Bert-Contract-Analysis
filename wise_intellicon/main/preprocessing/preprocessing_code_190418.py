import re

def special_letter_replacer(text):
    text = text.replace("'", '"').replace('“', '"').replace('”', '"').replace('‘', '"').replace('\t', ' ') \
            .replace('’', '"').replace('【','[').replace('】', ']').replace(']', ')').replace('　', ' ').replace('[', '(').replace('․', '.') \
            .replace('', ' 또는 ').replace('ᆞ', ' 또는 ').replace('∙', ' 또는 ').replace('･', ' 또는 ').replace('ㆍ', ' 또는 ').replace('·', ' 또는 ')  \
            .replace('～', '~')
    return text

def eng2kor(text):
    text = text.lower()
    text = text.replace('signature', '서명').replace('test', '테스트').replace('testing', '테스트').replace('know-how', '노하우') \
            .replace('version', '버전').replace('s/w', '소프트웨어').replace("sw", "소프트웨어").replace("software", "소프트웨어") \
            .replace("service", "서비스").replace('β', '베타').replace('db', '데이터베이스').replace('cd', '씨디').replace('a/s', '유지보수')
            
    return text

def synonym_replacer(text):
    text = text.replace('컨텐츠', '콘텐츠').replace('상호 협조', '상호협조').replace('귄리', '권리').replace('청구 일', '청구일').replace('제 3자', '제삼자') \
                .replace(' 소프트웨어', '프로그램')
    return text


def space_hander(text):
    text = re.sub('[ ]+', ' ', text)
    return text


# t = '계약당사자(갑, 을, 병)는, 갑 또는 을 사이에 추진될 을지로 병원 계약목적을 가진 갑이 작성함. 갑 회사 "갑" 회사 "갑"회사 갑의 회사'
def gap_eul_corrector(sentence):

    for subject in ['갑', '을', '병']:
        # re.sub함수는 title_pattern과 매치되는 sentence를 '"\\1"\\2'로 바꾼다. 
        
        if subject in sentence: # 갑, 을, 병) => _갑_, _을_, _병_)
            title_pattern = re.compile(r"([^가-힣])(["+subject+"])([^가-힣 ]){1}")   
             # 질병, 10일을, 이사람을) 이므로 
            sentence = re.sub(title_pattern, '\\1"\\2"\\3', sentence)
            
           
        if subject in sentence:
            sentence = sentence.replace('.'+subject+' ', '."'+subject+'" ')

        # 양 옆에 띄어쓰기가 있는 경우 명사 처리
        if subject in sentence:  
            sentence = sentence.replace(' '+subject+' ', ' "'+subject+'" ')
            
            if '등 "'+ subject + '" ' in sentence:
                sentence = sentence.replace(' "'+subject+'" ', ' '+subject+' ')
            elif ') "'+ subject + '" ' in sentence:
                sentence = sentence.replace(' "'+subject+'" ', ' '+subject+' ')
            elif '" "'+ subject + '" ' in sentence:
                sentence = sentence.replace(' "'+subject+'" ', ' '+subject+' ')
            
        # 맨 앞에 나오는 갑, 을, 병
        ###### 병 원 명 
        if subject in sentence:
            #title_pattern = re.compile(r"([^가-힣])(["+subject+"])([의이과에와으은을간사\,\)])")
            title_pattern = re.compile(r"([^가-힣])(["+subject+"])([의이과에와으은을간사로는자])")
            sentence = re.sub(title_pattern, '\\1"\\2"\\3', sentence)
            title_pattern_2 = re.compile(r"(^["+subject+"])([의이과에와으은을간사로는자 \:\"\,])")
            sentence = re.sub(title_pattern_2, '"\\1"\\2', sentence)

        #for josa in ['이라']:
        #    sentence = sentence.replace(' '+subject+josa, ' "'+subject+'"'+josa)
        
        # 갑 을 한 개씩 있는 문장은 무조건 명사 처리 
        if len(sentence)==1 and sentence==subject:
            sentence = sentence.replace(subject, '"'+subject+'"')

    sentence = sentence.replace('""', '"')

    return sentence


def word_connector(text):
    words = ['성명','직위','회사명','주소', '(갑)', '(을)', '(병)', '(인)', '연락처', '전화', '대표자', '제3자', '병원명', '계약명', '잔금']
    for w in words:
        text = text.replace(' '.join(list(w)), w)
    return text

def token_generator(text):
    words = [('"갑"', '_갑_'),
              ('"을"', '_을_'),
              ('"병"', '_병_'),
              ('" 갑 "', '_갑_'),
              ('"을 등"', '_을_등_'),

              ('(갑)', '_갑_'),
              ('(을)', '_을_'),
              ('(병)', '_병_'),

              ('[갑]', '_갑_'),
              ('[을]', '_을_'),
              ('[병]', '_병_'),

              ('(인)', '_인_'),
              ('제3자', '_제삼자_'),
              ('제삼자', '_제삼자_')]

    for before, after in words:
        text = text.replace(before, after)
    text = text.replace('__', '_')
    return text

def title_catcher(t):
    t = t.strip()
    title_pattern = re.compile("제[\s\d]+조[\s]{0,2}(\(|\[)?[\w\s,.;:\"\/\)\(]+(\)|\])?")
    if re.match(title_pattern, t):
        if t[-1] == '.':
            return False
        else:
            return True
    return False

def title_process(text):
    if title_catcher(preprocess(text)):
        # text = '_조제목_'
        if '(' in text and ')' in text:
            word = text[text.index('(')+1:text.index(')')]
            if len(re.sub('[ ]+', '', word))==2:
                word = re.sub('[ ]+', '', word)
                text = text[:text.index('(')+1]+word+text[text.index(')'):]
    return text

def preprocess(text):
    text = str(text)
    text = special_letter_replacer(text)
    text = space_hander(text)
    text = word_connector(text)
    text = eng2kor(text)
    text = synonym_replacer(text)    
    text = gap_eul_corrector(text)
    text = token_generator(text)
    text = space_hander(text)   
    return text

##################################################################################################################

def date_process(text):
    return re.sub('[0-9]{4}[ .년]{0,3}[0-9]{1,2}[ .월]{0,3}[0-9]{1,2}[ .일]{0,3}', '_날짜_', text)

def phone_process(text):
    return re.sub('[0-9]{2,4}[ -][0-9]{3,4}[ -][0-9]{4}', '_전화번호_', text)

def time_process(text):
    return re.sub('[0-9]{2}[: ][0-9]{2}', '_시간_', text)

##################################################################################################################

def software_postprocess(text, label):
    if re.match(r'.*(해지|종결).*제공한.*자료.*갑.에게.*반환.*다.', text):
        return ['비밀유지의무']
    elif '로열티' in text and text.strip()[-2:]=='다.':
        return label+['지식재산권']
    else:
        return label

def common_postprocess(label, prob, input_sentence):
    if label == ['PAD']:
        label = ['당사자']
    if sum(input_sentence)==0: 
        label = []
    if '계약일' in label and '준수기간' in label:
        label.remove('계약일')
    return label, 1

##################################################################################################################


def title_hander(t):
    title_pattern = re.compile(r"(^제[\s\d]+조)([\s]*(?:\(|\[|【)[\w\s,.;:\･\·\․\”\“\"\/]+(?:\)|\]|】))(.*)")
    t = re.sub(title_pattern, r"\1 \2\n\3", t)
    t = re.sub('[ ]+', ' ', t)
    if '\n' in t:
        n_idx = t.index('\n')+1
        if n_idx==len(t):
            t = t.replace('\n','')
    return t

def line_process(text):
    text = title_hander(text)
    return text

def paragraph_process(contract):
    contract = contract[0]
    if '\n' in contract:
        temp = [line_process(line) for line in contract.split('\n')]
    temp = []
    prev_line = ''
    for i, line in enumerate(contract):
        if prev_line != '' and re.match(r'단[.,].*.', line):
            temp.append(temp.pop()+ ' ' + line)
        elif prev_line != '' and re.match(r'다만[.,].*.', line):
            temp.append(temp.pop()+ ' ' + line)         
        else: 
            temp.append(line)
        prev_line = temp[-1]
    return [temp]

def contract_process(contract):
    return paragraph_process(contract) 

##################################################################################################################


def remove_duplicate_article_title(corpus):
    corpus_temp = []
    article_name = 'da39a3ee5e6b4b0d3255bfef95601890afd80709'
    for chunk in corpus:
        line_temp = []
        for line in chunk.split('\n'):
            if len(line) > 0 and line != '\t':
                temp = re.sub('제.{1,3}조', '_제N조_', line)
                if article_name == line:
                    pass
                else:
                    line_temp.append(line)

                if '_제N조_' in temp:
                    article_name = line
                        
        corpus_temp.append(line_temp)
    return corpus_temp