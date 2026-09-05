import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)

c = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ids = sorted(m.id for m in c.models.list())
print('total models:', len(ids))
for model_id in ids:
    print('model ', model_id)
    
# for want in ['gpt-4.1-mini','gpt-image-1-mini','gpt-4o-mini-tts']:
#     print(f"  {want}: {'AVAILABLE' if want in ids else 'NOT AVAILABLE'}")
print()
print('gpt-5/4.1 chat family visible:')
print(' ', [i for i in ids if i.startswith(('gpt-5','gpt-4.1')) and 'audio' not in i and 'chat' not in i][:14])
