import re
import gcopd.scienceworld.collection.state as st
from gcopd.scienceworld.environment import choices
def equivalent(a,b):
 ca,cb=choices(a),choices(b)
 if ca or cb:return sorted(ca.values())==sorted(cb.values()) and bool(ca) and bool(cb)
 return st.canonical_observation(a)==st.canonical_observation(b)
