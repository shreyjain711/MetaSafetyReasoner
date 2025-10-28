import json 

def get_json(response):
    if type(response) != str: return None
    try:
        return json.loads(response)
    except Exception as e:
        print(e)
        return None

def is_valid_response(response, validator, debug=True, do_lvl4_check=True):
    if validator == 'cat':
        resp = get_json(response)
        if resp is None:
            if debug: print("DBG-CV1", response)
            return False
        if 'category' not in resp: return False
        for k in ['code', 'name', 'subcategory', 'reason']: 
            if k not in resp['category']: 
                if debug: print("DBG-CV2", response)
                return False
        if resp['category']['code'] not in ['CIA', 'CSA', 'PDA', 'ELE', 'SSW', 'EGT', 'GEN']: 
            if debug: print("DBG-CV3", response)
            return False

        if do_lvl4_check:
            subcat_map = {
                'CIA': ['Illegal Sexual Content', 'Drug Crimes', 'Dangerous Goods', 'Property Violation', 'Economic Crimes', 'Human Trafficking', 'Illegal Gambling', 'Smuggling Crimes'],
                'CSA': ['Unauthorized Access', 'Malware', 'Data Destruction', 'Physical Penetration', 'Anonymous Abuse', 'AI Abuse', 'Cyber Espionage', 'Protocol Abuse'],
                'PDA': ['Privacy Breach', 'Abuse of Surveillance', 'Data Forgery', 'Corporate Espionage', 'Data Trafficking', 'Algorithm Abuse', 'Mass Surveillance'],
                'ELE': ['Non-Ethical Experiments', 'Self-Harm/Suicide Incitement', 'Evidence Destruction Techniques', 'Counter-Surveillance Techniques', 'Exploitation of Legal Loopholes', 'High-Risk Knowledge Dissemination'],
                'SSW': ['Dangerous Behavior Inducement', 'Psychological Manipulation and Exploitation', 'Incitement of Hatred and Violence', 'Spread of Extreme Ideologies', 'Dissemination of False Information', 'Incitement of Illegal Assemblies'],
                'EGT': ['Pollution Technology', 'Illegal Ecological Destruction', 'Pathogen Misuse', 'Biological Weapons Development', 'Climate Engineering Misuse', 'Weaponization of Disasters', 'Space Security Threats'],
                'GEN': ['Off-Topic Content', 'Semantically Unrelated', 'Irrelevant to Risk Analysis']
                }
            for code, subcats in subcat_map.items():
                if resp['category']['code'] == code and resp['category']['subcategory'] not in subcats:
                    if debug: print("DBG-CV4", response)
                    return False

    elif validator == 'sev':
        resp = get_json(response)
        if resp is None:
            print("DBG-SV1", response)
            return False
        
        if 'risk_level' not in resp: return False
        for k in ['score', 'label', 'reason']: 
            if k not in resp['risk_level']: 
                print("DBG-SV2", response)
                return False
        
        if resp['risk_level']['label'] not in ['Safe', 'Low Risk', 'Medium Risk', 'High Risk']: 
            print("DBG-SV3", response)
            return False

        lvl_scr_map = {
                'Safe': 0, 
                'Low Risk': 1, 
                'Medium Risk': 2, 
                'High Risk': 3
            }
        if lvl_scr_map[resp['risk_level']['label']] != int(resp['risk_level']['score']):
            print("DBG-SV4", response)
            return False
        
    return True
    