# =============================================
# ExamCraft FastAPI 백엔드
# =============================================

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import os
import json
from supabase import create_client, Client

# ── 환경변수 ──
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://tjtqibldbpajgowqbpim.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

app = FastAPI(title="ExamCraft API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase 클라이언트 (서비스 롤) ──
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ── 사용자 인증 ──
async def get_current_user(authorization: str = Header(...)):
    """JWT 토큰으로 사용자 인증"""
    try:
        token = authorization.replace("Bearer ", "")
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="인증 실패")
        return user.user
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"인증 오류: {str(e)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 학교 설정 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SchoolCreate(BaseModel):
    school_id: str
    school_name: str
    grade: Optional[str] = ""
    choice_style: Optional[str] = ""
    error_pattern: Optional[str] = ""
    stems: Optional[dict] = {}

@app.get("/schools")
async def get_schools(user=Depends(get_current_user)):
    res = supabase.table("schools").select("*").eq("user_id", user.id).execute()
    return res.data

@app.post("/schools")
async def create_school(data: SchoolCreate, user=Depends(get_current_user)):
    res = supabase.table("schools").upsert({
        "user_id": user.id,
        "school_id": data.school_id,
        "school_name": data.school_name,
        "grade": data.grade,
        "choice_style": data.choice_style,
        "error_pattern": data.error_pattern,
        "stems": data.stems,
    }, on_conflict="user_id,school_id").execute()
    return res.data

@app.put("/schools/{school_id}")
async def update_school(school_id: str, data: SchoolCreate, user=Depends(get_current_user)):
    res = supabase.table("schools").update({
        "school_name": data.school_name,
        "grade": data.grade,
        "choice_style": data.choice_style,
        "error_pattern": data.error_pattern,
        "stems": data.stems,
    }).eq("user_id", user.id).eq("school_id", school_id).execute()
    return res.data

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유형 템플릿 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TemplateItem(BaseModel):
    type_code: str
    prompt: Optional[str] = ""

@app.get("/templates/{school_id}")
async def get_templates(school_id: str, user=Depends(get_current_user)):
    res = supabase.table("templates").select("*")\
        .eq("user_id", user.id).eq("school_id", school_id).execute()
    return res.data

@app.post("/templates/{school_id}")
async def save_templates(school_id: str, items: List[TemplateItem], user=Depends(get_current_user)):
    rows = [{
        "user_id": user.id,
        "school_id": school_id,
        "type_code": item.type_code,
        "prompt": item.prompt or "",
    } for item in items]
    res = supabase.table("templates").upsert(rows, on_conflict="user_id,school_id,type_code").execute()
    return {"success": True}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 문제 생성 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GenerateRequest(BaseModel):
    passage: str
    type_codes: List[str]
    school_id: str
    passage_id: Optional[str] = ""

CREDIT_PER_QUESTION = 1  # 문제 1개당 크레딧 1개 차감

def get_stem(stems: dict, type_code: str) -> str:
    stem_map = {
        "MT01": "발문_빈칸", "MT02": "발문_어법오류문장",
        "MT03": "발문_대화문빈칸", "MT04": "발문_어휘문맥",
        "MT05": "발문_순서배열", "MT06": "발문_문장삽입",
        "MT07": "발문_제목", "MT08": "발문_요지",
        "MT09": "발문_내용일치", "MT10": "발문_내용불일치",
        "MT11": "발문_영영풀이", "MT12": "발문_지칭추론",
        "MT13": "발문_어법밑줄", "MT14": "발문_요약빈칸",
        "MT15": "발문_서답형",
    }
    key = stem_map.get(type_code, "")
    return stems.get(key, "")

async def call_claude(system_prompt: str, user_prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 2000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        )
        data = res.json()
        if "error" in data:
            raise Exception(data["error"].get("message", "Claude API 오류"))
        return data["content"][0]["text"]

def build_system_prompt():
    return """당신은 대한민국 중학교 영어 내신 전문 교재 제작자입니다.
모든 문제는 중1~3 수준에 맞게 출제하라. 강남권 중학교 내신 기준 중~상 난이도로 출제하라.

[필수 출력 규칙]
1. 빈칸 표시: 빈칸은 반드시 ______ (언더바 6개)로만 표시하라. 빈칸 자리에 절대 답을 채우지 마라.
2. 밑줄 표시: 밑줄 칠 어구는 _어구_ 형식으로 표시하라. (언더바로 감싸기)
   - 번호기호가 있으면: ⓐ_어구_ ①_어구_ 형식
   - 번호기호가 없으면: _어구_ 형식
   - 언더바 6개 이상(______)은 빈칸 표시이므로 밑줄 표시에 사용하지 마라.
3. question 필드에서 발문을 포함하지 마라. question 필드는 지문 내용만 포함하라.
4. 선택지(choices) 필드: 선택지 5개를 반드시 배열의 개별 요소로 분리하라.
   올바른 예: ["①문장1", "②문장2", "③문장3", "④문장4", "⑤문장5"]
   잘못된 예: "①문장1 ②문장2 ③문장3" (하나로 합치기 금지)
5. 서답형(MT15)은 choices를 빈 배열 []로 반환.
6. question/answer 필드에서 큰따옴표 사용 금지. 반드시 작은따옴표 사용.

출력은 반드시 { 로 시작해서 } 로 끝나는 순수 JSON 객체만 출력하세요.
{
  "question": "지문 내용",
  "choices": ["①선택지1", "②선택지2", "③선택지3", "④선택지4", "⑤선택지5"],
  "answer": "정답 번호 및 내용",
  "explanation": "해설"
}"""

@app.post("/generate")
async def generate_questions(req: GenerateRequest, user=Depends(get_current_user)):
    # 1. 크레딧 확인
    credit_res = supabase.table("credits").select("balance").eq("user_id", user.id).execute()
    if not credit_res.data:
        # 크레딧 레코드 없으면 생성
        supabase.table("credits").insert({"user_id": user.id, "balance": 0}).execute()
        balance = 0
    else:
        balance = credit_res.data[0]["balance"]

    required = len(req.type_codes) * CREDIT_PER_QUESTION
    if balance < required:
        raise HTTPException(status_code=402, detail=f"크레딧 부족 (필요: {required}, 보유: {balance})")

    # 2. 학교/템플릿/발문 로드
    school_res = supabase.table("schools").select("*")\
        .eq("user_id", user.id).eq("school_id", req.school_id).execute()
    school = school_res.data[0] if school_res.data else {}
    stems = school.get("stems", {})

    tmpl_res = supabase.table("templates").select("*")\
        .eq("user_id", user.id).eq("school_id", req.school_id).execute()
    template_map = {t["type_code"]: t["prompt"] for t in (tmpl_res.data or [])}

    # 3. 지문 저장
    if req.passage_id:
        supabase.table("passages").upsert({
            "user_id": user.id,
            "school_id": req.school_id,
            "passage_id": req.passage_id,
            "content": req.passage,
        }, on_conflict="user_id,school_id,passage_id").execute()

    # 4. 문제 생성
    results = []
    system_prompt = build_system_prompt()

    for type_code in req.type_codes:
        stem = get_stem(stems, type_code)
        prompt = template_map.get(type_code, "")
        stem_line = f'이 학교의 발문: "{stem}" — 이 발문을 그대로 question 첫 줄에 사용하라.\n\n' if stem else ""
        user_prompt = f"[{type_code}] 유형 문제를 출제하세요.\n\n{stem_line}출제 지침:\n{prompt}\n\n지문/입력:\n{req.passage}\n\n반드시 JSON만 출력하세요."

        try:
            raw = await call_claude(system_prompt, user_prompt)
            clean = raw.replace("```json", "").replace("```", "").strip()
            start = clean.find("{")
            end = clean.rfind("}")
            if start != -1 and end != -1:
                clean = clean[start:end+1]
            parsed = json.loads(clean)

            # 5. DB 저장
            q_res = supabase.table("questions").insert({
                "user_id": user.id,
                "school_id": req.school_id,
                "passage_id": req.passage_id or "",
                "type_code": type_code,
                "stem": stem or parsed.get("question", "").split("\n")[0],
                "question": parsed.get("question", ""),
                "choices": parsed.get("choices", []),
                "answer": parsed.get("answer", ""),
                "explanation": parsed.get("explanation", ""),
            }).execute()

            results.append({
                "success": True,
                "type_code": type_code,
                "id": q_res.data[0]["id"] if q_res.data else None,
                **parsed
            })

            # 6. 크레딧 차감
            supabase.table("credits").update({"balance": balance - CREDIT_PER_QUESTION})\
                .eq("user_id", user.id).execute()
            supabase.table("credit_history").insert({
                "user_id": user.id,
                "amount": -CREDIT_PER_QUESTION,
                "type": "use",
                "description": f"{req.school_id} {type_code} 문제 생성",
            }).execute()
            balance -= CREDIT_PER_QUESTION

        except Exception as e:
            results.append({"success": False, "type_code": type_code, "error": str(e)})

    return {"results": results}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 문제 조회/수정 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/questions")
async def get_questions(school_id: Optional[str] = None, user=Depends(get_current_user)):
    q = supabase.table("questions").select("*").eq("user_id", user.id)
    if school_id:
        q = q.eq("school_id", school_id)
    res = q.order("created_at", desc=True).execute()
    return res.data

class QuestionUpdate(BaseModel):
    stem: Optional[str] = None
    question: Optional[str] = None
    choices: Optional[list] = None
    answer: Optional[str] = None
    explanation: Optional[str] = None
    status: Optional[str] = None
    memo: Optional[str] = None

@app.put("/questions/{question_id}")
async def update_question(question_id: str, data: QuestionUpdate, user=Depends(get_current_user)):
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    res = supabase.table("questions").update(update_data)\
        .eq("id", question_id).eq("user_id", user.id).execute()
    return res.data

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 재출제 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post("/questions/{question_id}/regenerate")
async def regenerate_question(question_id: str, user=Depends(get_current_user)):
    # 기존 문제 조회
    q_res = supabase.table("questions").select("*")\
        .eq("id", question_id).eq("user_id", user.id).execute()
    if not q_res.data:
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다")

    q = q_res.data[0]
    school_id = q["school_id"]
    passage_id = q["passage_id"]
    type_code = q["type_code"]

    # 저장된 지문 조회
    p_res = supabase.table("passages").select("content")\
        .eq("user_id", user.id).eq("school_id", school_id).eq("passage_id", passage_id).execute()
    if not p_res.data:
        raise HTTPException(status_code=404, detail="저장된 지문을 찾을 수 없습니다")

    passage = p_res.data[0]["content"]

    # 재생성 (generate와 동일 로직)
    gen_res = await generate_questions(
        GenerateRequest(
            passage=passage,
            type_codes=[type_code],
            school_id=school_id,
            passage_id=passage_id,
        ),
        user
    )

    if gen_res["results"] and gen_res["results"][0].get("success"):
        new_q = gen_res["results"][0]
        # 기존 문제 덮어쓰기
        supabase.table("questions").update({
            "question": new_q.get("question", ""),
            "choices": new_q.get("choices", []),
            "answer": new_q.get("answer", ""),
            "explanation": new_q.get("explanation", ""),
            "status": "미검토",
            "memo": "",
        }).eq("id", question_id).execute()
        # 새로 생성된 중복 레코드 삭제
        if new_q.get("id"):
            supabase.table("questions").delete().eq("id", new_q["id"]).execute()

    return {"success": True}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 크레딧 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/credits")
async def get_credits(user=Depends(get_current_user)):
    res = supabase.table("credits").select("balance").eq("user_id", user.id).execute()
    if not res.data:
        supabase.table("credits").insert({"user_id": user.id, "balance": 0}).execute()
        return {"balance": 0}
    return {"balance": res.data[0]["balance"]}

@app.get("/credits/history")
async def get_credit_history(user=Depends(get_current_user)):
    res = supabase.table("credit_history").select("*")\
        .eq("user_id", user.id).order("created_at", desc=True).limit(50).execute()
    return res.data

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 헬스체크
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
