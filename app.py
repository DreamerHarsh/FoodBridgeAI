import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, asin, sqrt

st.set_page_config(page_title="FoodBridge AI", page_icon="🥗", layout="wide")
DONATIONS = [
 {"id":"FD-101","provider":"Campus Hostel","food_type":"cooked vegetarian meals rice dal vegetables","quantity":45,"unit":"meals","city":"Phagwara","lat":31.224,"lon":75.770,"hours_left":3,"status":"Available"},
 {"id":"FD-102","provider":"Green Bowl Cafe","food_type":"bread bakery sandwiches","quantity":25,"unit":"portions","city":"Phagwara","lat":31.218,"lon":75.775,"hours_left":5,"status":"Available"},
 {"id":"FD-103","provider":"Community Event","food_type":"packaged snacks fruit juice","quantity":60,"unit":"packs","city":"Phagwara","lat":31.230,"lon":75.765,"hours_left":6,"status":"Available"}]
RECIPIENTS = [
 {"name":"Hope Community Kitchen","needs":"cooked vegetarian meals rice dal vegetables","capacity":60,"lat":31.226,"lon":75.768,"contact":"demo-hope@example.org"},
 {"name":"Helping Hands Shelter","needs":"bread bakery sandwiches cooked meals","capacity":30,"lat":31.215,"lon":75.780,"contact":"demo-hands@example.org"},
 {"name":"Youth Relief Centre","needs":"packaged snacks fruit juice dry food","capacity":80,"lat":31.232,"lon":75.760,"contact":"demo-youth@example.org"},
 {"name":"Neighbourhood Support Group","needs":"vegetarian meals bread fruit snacks","capacity":20,"lat":31.220,"lon":75.768,"contact":"demo-neighbour@example.org"}]
if "donations" not in st.session_state: st.session_state.donations = pd.DataFrame(DONATIONS)
if "recipients" not in st.session_state: st.session_state.recipients = pd.DataFrame(RECIPIENTS)
if "matches" not in st.session_state: st.session_state.matches = pd.DataFrame()
def distance_km(a,b,c,d):
    a,b,c,d=map(radians,[a,b,c,d]); dl, da=d-b, c-a
    x=sin(da/2)**2+cos(a)*cos(c)*sin(dl/2)**2
    return 6371*2*asin(sqrt(x))
def match(d, rs):
    docs=[str(d.food_type)]+rs.needs.fillna("").astype(str).tolist()
    try: sims=cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(docs)[0:1],TfidfVectorizer(stop_words="english").fit_transform(docs)[1:]).flatten()
    except Exception:
        # Fit once correctly if the token vocabulary is small or empty
        mat=TfidfVectorizer().fit_transform(docs); sims=cosine_similarity(mat[0:1],mat[1:]).flatten()
    out=[]
    for i,(_,r) in enumerate(rs.iterrows()):
        dist=distance_km(float(d.lat),float(d.lon),float(r.lat),float(r.lon))
        cap=min(float(r.capacity)/max(float(d.quantity),1),1)
        near=max(0,1-dist/15); urgent=max(0,1-float(d.hours_left)/24)
        score=100*(.50*float(sims[i])+.25*cap+.15*near+.10*urgent)
        out.append({"Recipient":r["name"],"Match score":round(score,1),"Text relevance (%)":round(float(sims[i])*100,1),"Distance (km)":round(dist,2),"Capacity":int(r.capacity),"Demo contact":r.contact})
    return pd.DataFrame(out).sort_values("Match score",ascending=False).reset_index(drop=True)

st.title("🥗 FoodBridge AI")
st.subheader("AI-assisted surplus-food matching and pickup coordination")
st.caption("SDG 2 — Zero Hunger | Prototype uses synthetic demo records; it does not arrange real food pickups.")
page=st.sidebar.radio("Navigate",["Dashboard","Add donation","Find matches","Workflow & export"])
don=st.session_state.donations
rec=st.session_state.recipients

if page=="Dashboard":
    a,b,c,e=st.columns(4)
    a.metric("Demo donations",len(don)); b.metric("Available",int((don.status=="Available").sum()))
    c.metric("Potential recipients",len(rec)); e.metric("Food units listed",int(don.quantity.sum()))
    st.markdown("### Current listings")
    st.dataframe(don[["id","provider","food_type","quantity","unit","city","hours_left","status"]],use_container_width=True,hide_index=True)
    st.markdown("### Matching formula")
    st.write("50% TF-IDF text similarity + 25% capacity fit + 15% distance + 10% pickup urgency.")
elif page=="Add donation":
    st.markdown("### Add surplus food")
    with st.form("add",clear_on_submit=True):
        provider=st.text_input("Provider","My College Canteen")
        food=st.text_input("Food description","cooked vegetarian meals rice dal")
        qty=st.number_input("Quantity",1,10000,20)
        unit=st.selectbox("Unit",["meals","portions","packs","kg","boxes"])
        city=st.text_input("City","Phagwara")
        x,y=st.columns(2); lat=x.number_input("Latitude",value=31.224,format="%.6f"); lon=y.number_input("Longitude",value=75.770,format="%.6f")
        hrs=st.slider("Hours until pickup deadline",1,24,4)
        ok=st.form_submit_button("Add listing")
    if ok:
        row={"id":f"FD-{101+len(don)}","provider":provider,"food_type":food,"quantity":int(qty),"unit":unit,"city":city,"lat":lat,"lon":lon,"hours_left":hrs,"status":"Available"}
        st.session_state.donations=pd.concat([don,pd.DataFrame([row])],ignore_index=True)
        st.success("Listing added. Open Find matches.")
elif page=="Find matches":
    avail=don[don.status=="Available"]
    if avail.empty: st.info("No available listings. Add one first.")
    else:
        labels={f'{r["id"]} — {r["provider"]} ({r["quantity"]} {r["unit"]})':r["id"] for _,r in avail.iterrows()}
        label=st.selectbox("Choose donation",list(labels))
        d=avail[avail.id==labels[label]].iloc[0]
        if st.button("Run matching agent",type="primary"):
            st.session_state.matches=match(d,rec); st.session_state.selected_id=d["id"]
        if not st.session_state.matches.empty and st.session_state.get("selected_id")==d["id"]:
            st.dataframe(st.session_state.matches,use_container_width=True,hide_index=True)
            best=st.session_state.matches.iloc[0]
            st.success(f'Top recommendation: {best["Recipient"]} — {best["Match score"]}/100')
            st.text_area("Draft outreach (not sent)",f'Hello {best["Recipient"]}, FoodBridge AI identified a possible match for {int(d.quantity)} {d.unit} of {d.food_type} from {d.provider} in {d.city}. Pickup deadline is within {int(d.hours_left)} hours. Please verify food safety, capacity and pickup availability. This is a draft only; no message has been sent.',height=130)
            st.caption("Human verification is required. Similarity and match scores are ranking signals, not probabilities.")
else:
    st.markdown("### Update listing status")
    labels={f'{r["id"]} — {r["provider"]}':r["id"] for _,r in don.iterrows()}
    label=st.selectbox("Donation",list(labels)); did=labels[label]
    current=don.loc[don.id==did,"status"].iloc[0]
    states=["Available","Recipient contacted (demo)","Pickup confirmed (demo)","Completed (demo)","Cancelled (demo)"]
    status=st.selectbox("Status",states,index=states.index(current) if current in states else 0)
    if st.button("Save status"):
        st.session_state.donations.loc[st.session_state.donations.id==did,"status"]=status
        st.success("Updated for this session.")
    st.download_button("Download donations CSV",st.session_state.donations.to_csv(index=False).encode(), "foodbridge_donations.csv","text/csv")
    if not st.session_state.matches.empty:
        st.download_button("Download match results CSV",st.session_state.matches.to_csv(index=False).encode(),"foodbridge_matches.csv","text/csv")
    st.markdown("### Limitations")
    st.write("Demo contacts and coordinates are synthetic. No real messages or pickups occur. A real pilot needs verified organisations, consent, food-safety checks, secure data handling and genuine notification integrations.")
st.divider()
st.caption("FoodBridge AI | Student prototype | SDG 2: Zero Hunger")
