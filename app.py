# FULL VERSION C + LLM (Gemini 2.0 Flash, Option 3, selected% stored as float full precision)
import re, math, json
from io import BytesIO
import pandas as pd
import streamlit as st

API_KEY = "WEJKEBHABLRKJVBR;KEARVBBVEKJ"

GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HbA1c & Weight % Reduction Extractor", layout="wide")
st.title("HbA1c / A1c + Weight — Regex Extraction + Gemini 2.0 Flash")

NUM = r'(?:[+-]?\d+(?:[.,·]\d+)?)'
PCT = rf'({NUM})\s*%'
DASH = r'(?:-|–|—)'
FROM_TO   = rf'from\s+({NUM})\s*%\s*(?:to|->|{DASH})\s*({NUM})\s*%'
REDUCE_BY = rf'(?:reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\s*(?:by\s*)?({NUM})\s*%'
ABS_PP    = rf'(?:absolute\s+reduction\s+of|reduction\s+of)\s*({NUM})\s*%'
RANGE_PCT = rf'({NUM})\s*{DASH}\s*({NUM})\s*%'
FLAGS = re.IGNORECASE
re_pct       = re.compile(PCT, FLAGS)
re_fromto    = re.compile(FROM_TO, FLAGS)
re_reduce_by = re.compile(REDUCE_BY, FLAGS)
re_abs_pp    = re.compile(ABS_PP, FLAGS)
re_range     = re.compile(RANGE_PCT, FLAGS)

re_hba1c  = re.compile(r'\bhb\s*a1c\b|\bhba1c\b|\ba1c\b', FLAGS)
re_weight = re.compile(r'\b(body\s*weight|weight|bw)\b', FLAGS)
re_reduction_cue = re.compile(r'\b(from|reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|drop(?:ped)?|fell|lower(?:ed|ing)?|declin(?:e|ed|ing))\b', FLAGS)


def parse_number(s):
    if s is None:return float('nan')
    s=str(s).replace(',', '.').replace('·','.').strip()
    try:return float(s)
    except:return float('nan')

def split_sents(t):
    if not isinstance(t,str):return []
    t=t.replace('\r','\n')
    parts=re.split(r'(?<=[\.\!\?])\s+|\n+',t)
    return [p.strip() for p in parts if p.strip()]

def sentence_ok(s,term):
    return bool(term.search(s) and re_pct.search(s) and re_reduction_cue.search(s))

def fmt_pct(v):
    if v is None or (isinstance(v,float) and math.isnan(v)):return ''
    return (f"{v:.3f}".rstrip('0').rstrip('.')+'%')

def window_pm5(s,pos):
    space={' ','\t','\n','\r'};L=len(s)
    i=pos-1;sp=0;LBS=pos
    while i>=0 and sp<5:
        if s[i] in space:
            while i>=0 and s[i] in space:i-=1
            sp+=1;LBS=i+1
        else:i-=1
    j=LBS-1
    while j>=0 and s[j] not in space:j-=1
    start=j+1
    k=pos;sp=0;RBE=pos
    while k<L and sp<5:
        if s[k] in space:
            while k<L and s[k] in space:k+=1
            sp+=1;RBE=k
        else:k+=1
    m=RBE
    while m<L and s[m] not in space:m+=1
    end=m
    start=max(0,start);end=max(start,min(end,L))
    return s[start:end],start

def add_match(out,si,abs_s,m,typ,vals,red):
    out.append({'raw':m.group(0),'type':typ,'values':vals,'reduction_pp':red,'si':si,'span':(abs_s+m.start(),abs_s+m.end())})

def extract_in_sentence(sent,si,term,tag):
    out=[];hit=False
    for m in re_fromto.finditer(sent):
        a=parse_number(m.group(1));b=parse_number(m.group(2));r=a-b if not math.isnan(a) and not math.isnan(b) else None
        add_match(out,si,0,m,f'{tag}:from-to',[a,b],r)
    for hh in term.finditer(sent):
        seg,base=window_pm5(sent,hh.end())
        for m in re_reduce_by.finditer(seg):v=parse_number(m.group(1));hit=True;add_match(out,si,base,m,f'{tag}:reduce_by',[v],v)
        for m in re_abs_pp.finditer(seg):v=parse_number(m.group(1));hit=True;add_match(out,si,base,m,f'{tag}:abs_pp',[v],v)
        for m in re_range.finditer(seg):a=parse_number(m.group(1));b=parse_number(m.group(2));rep=max(a,b);hit=True;add_match(out,si,base,m,f'{tag}:range',[a,b],rep)
        for m in re_pct.finditer(seg):v=parse_number(m.group(1));hit=True;add_match(out,si,base,m,f'{tag}:pct',[v],v)
    if tag=='weight' and not hit:
        for hh in term.finditer(sent):
            pos=hh.start();left=max(0,pos-60);chunk=sent[left:pos];last=None
            for m in re_pct.finditer(chunk):last=m
            if last is not None:v=parse_number(last.group(1));add_match(out,si,left,last,f'{tag}:prev60',[v],v)
    seen=set();res=[]
    for mm in out:
        sp=mm['span']
        if sp in seen:continue
        seen.add(sp);res.append(mm)
    res.sort(key=lambda x:x['span'][0])
    return res

def extract_text(text,term,tag):
    matches=[];sents=[]
    for si,s in enumerate(split_sents(text)):
        if not sentence_ok(s,term):continue
        sents.append(s)
        matches.extend(extract_in_sentence(s,si,term,tag))
    seen=set();res=[]
    for m in matches:
        k=(m['si'],m['span'])
        if k in seen:continue
        seen.add(k);res.append(m)
    res.sort(key=lambda x:(x['si'],x['span'][0]))
    return res,sents

# LLM

def configure_gemini():
    if not GENAI_AVAILABLE or not API_KEY:return None
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel("gemini-2.0-flash")
    except:return None

LLM_SYS=(
"Extract the primary % change for the target (HbA1c or weight).\n"
"You may select a value not explicitly listed if context implies it.\n"
"Return JSON: {\"extracted\": [...], \"selected_percent\": \"X%\"}.\n"
)

def norm_percent(v):
    if not v:return ''
    v=str(v).strip().replace(' ','')
    if v and not v.endswith('%') and re.match(r'^[+-]?\d+(?:[.,·]\d+)?$',v):v+='%'

def llm_pick(model, sentences, extracted_list):
    if not model or not sentences:
        return extracted_list, None
    prompt = LLM_SYS + "
TEXT:
" + " | ".join(sentences) + "
RegexExtracted:
" + json.dumps(extracted_list)
    try:
        resp = model.generate_content(prompt)
        txt = resp.text
        data = json.loads(txt)
        llm_vals = data.get("extracted", [])
        sel = data.get("selected_percent", "")
        sel = norm_percent(sel)
        # S2: absolute numeric
        try:
            num = parse_number(sel.replace('%',''))
            if not math.isnan(num): num = abs(num)
        except:
            num = None
        return llm_vals, num
    except:
        return extracted_list, None


# --- Main processing ---
@st.cache_data
def process_df(df, text_col):
    model = configure_gemini()
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col, '') or ''

        hba_matches, hba_sent = extract_text(text, re_hba1c, 'hba1c')
        # keep only <7 rule
        hba_matches = [m for m in hba_matches if (m['reduction_pp'] and abs(m['reduction_pp']) < 7) or any(abs(parse_number(v))<7 for v in m['values'])]
        hba_extracted = [fmt_pct(m['reduction_pp']) if 'from-to' in m['type'] else m['raw'] for m in hba_matches]

        wt_matches, wt_sent = extract_text(text, re_weight, 'weight')
        wt_extracted = [fmt_pct(m['reduction_pp']) if 'from-to' in m['type'] else m['raw'] for m in wt_matches]

        # LLM picks
        hba_llm_list, hba_sel = llm_pick(model, hba_sent, hba_extracted)
        wt_llm_list, wt_sel = llm_pick(model, wt_sent, wt_extracted)

        new = row.to_dict()
        new.update({
            'sentence': ' | '.join(hba_sent),
            'extracted_matches': hba_extracted,
            'LLM extracted': hba_llm_list,
            'selected %': hba_sel,
            'weight_sentence': ' | '.join(wt_sent),
            'weight_extracted_matches': wt_extracted,
            'weight LLM extracted': wt_llm_list,
            'weight selected %': wt_sel,
        })
        rows.append(new)

    out = pd.DataFrame(rows)
    def ok(x): return isinstance(x,list) and len(x)>0
    mask_h = out['extracted_matches'].apply(ok)
    mask_w = out['weight_extracted_matches'].apply(ok)
    out = out[mask_h | mask_w].reset_index(drop=True)
    return out


uploaded   = st.sidebar.file_uploader('Upload Excel (.xlsx) or CSV', type=['xlsx', 'xls', 'csv'])
col_name   = st.sidebar.text_input('Column with abstracts/text', value='abstract')
sheet_name = st.sidebar.text_input('Excel sheet name (blank = first sheet)', value='')

if not uploaded:
    st.stop()

try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, sheet_name=sheet_name or None)
except Exception as e:
    st.error(e)
    st.stop()

out_df = process_df(df, col_name)
st.dataframe(out_df.head(200))

@st.cache_data
def to_xlsx(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf) as w:
        df.to_excel(w, index=False)
    buf.seek(0)
    return buf
st.download_button("Download", to_xlsx(out_df), "results.xlsx")
