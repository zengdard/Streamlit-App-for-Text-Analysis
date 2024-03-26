import streamlit as st
from pathlib import Path


from st_on_hover_tabs import on_hover_tabs

from web_app_1 import lauchn
from streamlit_lottie import st_lottie

import requests

from validate_email import validate_email

import stripe


placeholder = st.empty()


THIS_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
STYLES_DIR = THIS_DIR / "STYLE"
CSS_FILE = STYLES_DIR / "style.css"


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
def is_user_connected():
    return st.session_state.get('connected', False)

def login():
    # Définir une variable de session nommée "connected" avec une valeur de "True"
    st.write('Connecté !')
    st.session_state['connected'] = True

def logout():
    # Supprimer la variable de session "connected"
    st.write('Déconnecté !')
    st.session_state.pop('connected', None)
    connexion_session()

# Fonction pour vérifier si l'utilisateur est connecté ou non

def connexion_session():
           
            load_css_file(CSS_FILE)

            with st.sidebar:
                tab = on_hover_tabs(tabName=['Se connecter','S\'abonner à +'], 
                                    iconName=['dashboard','home'],
                                    styles = {'navtab': {'background-color':'#FFFFFF',
                                                        'color': '#000000',
                                                        'font-size': '18px',
                                                        'transition': '.3s',
                                                        'white-space': 'nowrap',
                                                        'text-transform': 'uppercase'},
                                            'tabOptionsStyle': {':hover :hover': {'color': 'red',
                                                                            'cursor': 'pointer'}},
                                            'iconStyle':{'position':'fixed',
                                                            'left':'7.5px',
                                                            'text-align': 'left'},
                                            'tabStyle' : {'list-style-type': 'none',
                                                            'margin-bottom': '30px',
                                                            'padding-left': '30px'}},
                                    key="1")

            if tab == 'Se connecter' :
                    

                    # Récupérer la liste des utilisateurs Stripe

                    accepted_terms = st.checkbox("J'accepte les termes et conditions d'utilisation de  .")

                    if st.button('Se connecter à ') :
                        if validate_email(_email)   :
                            
                            if accepted_terms : 
                                utilisateurs = stripe.Customer.list()

                            
                                for user in utilisateurs["data"] :
                                    if _email == user['email']:
                                            log  = True
                                            break
                                    else:
                                            st.warning('Nous ne trouvons pas votre addresse mail.')
                                            log = False

                                if log:
                                        
                                        st.session_state['connected'] = True
                                        lauchn()

                            else :
                                st.warning('Veuillez accepter les termes et conditions.')

                        else:
                            st.warning('Addresse mail non valide. Si vous rencontrez des problèmes de connexion vous pouvez nous contacter à cette addresse contact@ .fr')

            elif tab == 'S\'abonner à  +':
                    st.title(" +")
                    #left_col, right_col = st.columns((2, 1))
                    
                
                    st.subheader('Accédez à l\'entièreté de   via  +')




                    
                    animation_url = "https://assets8.lottiefiles.com/packages/lf20_ko8ud57v.json"
                    lottie_json = load_lottieurl(animation_url)
                    st_lottie(lottie_json)

                    features = {
                    "https://assets9.lottiefiles.com/private_files/lf30_ydctjlvt.json": [
                        "Vérification d'Images",
                        "Les étudiants ne se sont pas fait attendre pour utiliser les générateurs de texte dopés à l'intelligence artificielle pour frauduleusement soumettre leurs devoirs."
                        ],
                    "https://assets3.lottiefiles.com/packages/lf20_7fwvvesa.json": [
                        "Vérification de Textes",
                        "Tout comme la génération de textes, la génération d'images a vu  apparaître l'émergence de nombreuses désinformations à travers les réseaux sociaux et les médias."
                          ],
                    "https://assets6.lottiefiles.com/packages/lf20_xbf1be8x.json": [
                        "Vérification de Sources",
                        "Méfiez-vous des articles et des contenus générés par des intelligences artificielles à votre insu, en certifiant vos lectures par  ."
                              ],
                    }
                    for image, description in features.items():
                        lottie_json = load_lottieurl(image)
                        st.write("")
                        left_col, right_col = st.columns(2)
                        
                        right_col.write(f"**{description[0]}**")
                        right_col.write(description[1])
                        with left_col:
                            st_lottie(lottie_json, key=description[0])

                    right_col.write("")
                    
                    right_col.write("")
                    
                    right_col.write("")
                    
                    right_col.write("")
                    right_col.markdown(f"<a href={'https://buy.stripe.com/bIYeVe2GP1DF9BSdQQ'} class='button'>S'abonner</a>",
                    unsafe_allow_html=True)


                     
                            
                   # with st.form(key='my_form'):
                    #    email2 = st.text_input('Insérez votre email ici.', '')
                     #   NAME = st.text_input('Insérez votre nom/pseudo ici.', '')
                      #  new_password = st.text_input("Password",type='password')
                       # new_password2 = st.text_input("Réécrivez votre password",type='password')
                        #accepted_terms = st.checkbox("J'accepte les termes et conditions")


                        #submit_button = st.form_submit_button(label='Submit')

                    #if submit_button and  validate_email.validate_email(email2):
                    #    if accepted_terms : 
                    #                    print(utilisateurs)
                    #            
                    #                    if new_password != new_password2 : 
                    #                        st.warning('Mauvais mot de passe.')
#
#
                    #                    else :
#
                    #                        creat_account(make_hashes(email2),make_hashes(new_password), KEY_STRIPE, NAME, DATE_KEY)
                    #                        st.success("You have successfully created a valid Account")
                    #                        st.info("Go to Login Menu to login")
                    
                            #else:
                            #     st.warning('Nous trouvons aucunes adresses associées, vérifiez l\'orthographe, ou patientez pendant la validation.')
                  #      else :
                  #          st.warning('Veuillez accetpter les termes et conditions.')
                  #  else :
                  #       st.warning('Vérifiez votre adresse mail.')




# Afficher la page de l'application uniquement si l'utilisateur est connecté
if is_user_connected():
    lauchn()
    # Reste du code de l'application Streamlit
else:
    connexion_session()
