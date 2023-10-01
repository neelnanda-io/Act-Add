# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("pythia-1b", device="cuda")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
pos_prompt = " weddings"
neg_prompt = " "
# pos_prompt = "I talk about weddings constantly"
# pos_prompt = torch.cat([model.to_tokens(pos_prompt)[0], torch.tensor([model.to_single_token(" ")]*2).to("cuda")])
# neg_prompt = "I do not talk about weddings constantly"
gen_prompt = "I went up to my friend and said"
print(len(model.to_str_tokens(pos_prompt)))
print(len(model.to_str_tokens(neg_prompt)))
print(len(model.to_str_tokens(gen_prompt)))
# %%
model.reset_hooks(including_permanent=True)
coef = 10
layer = 3
pos_logits, pos_cache = model.run_with_cache(pos_prompt, prepend_bos=True)
pos_resids = pos_cache["resid_post", layer][0]
neg_logits, neg_cache = model.run_with_cache(neg_prompt, prepend_bos=True)
neg_resids = neg_cache["resid_post", layer][0]
resid_diff = (pos_resids - neg_resids)[-1]
print(resid_diff.shape, pos_resids.shape, neg_resids.shape)
USE_ACT_ADD = [True]
POS_INTERVENE = [0]
def act_add_hook(resid_post, hook):
    
    if USE_ACT_ADD[0] and len(resid_post[0])>1:
        # print(resid_post.shape)
        resid_post[0, POS_INTERVENE[0], :] += coef * resid_diff
    return resid_post
model.reset_hooks(including_permanent=True)
model.blocks[layer].hook_resid_post.add_perma_hook(act_add_hook)
scores = []
for p in range(9):
    POS_INTERVENE[0] = p
    strings = []
    wedding_words = ["wedding", "marr"]
    for i in range(5):
        s = (model.generate(gen_prompt, 30, top_p=0.3, freq_penalty=1.0, temperature=1.0, prepend_bos=True, verbose=False))
        if i==0: print(s)
        strings.append(s)
    is_wedding = [any(word in s for word in wedding_words) for s in strings]
    print(is_wedding)
    print(sum(is_wedding)/len(is_wedding))
    scores.append(sum(is_wedding)/len(is_wedding))
print(scores)
# %%

# %%
