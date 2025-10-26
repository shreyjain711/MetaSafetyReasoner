import json 

def is_valid_response(response, validator):
    if validator == 'risk_category':
        if len(response) < 2: return False
        if type(response[0]) != int: return False
        if type(response[1]) != str: return False
        try:
            resp = json.loads(response[1])
        except:
            return False
        if 'category' not in resp: return False
        for k in ['code', 'name', 'subcategory', 'reason']: 
            if k not in resp['category']: 
                return False
        if resp['category']['code'] not in ['CIA', 'CSA', 'PDA', 'ELE', 'SSW', 'EGT', 'GEN']: return False
        if resp['category']['code'] not in ['CIA', 'CSA', 'PDA', 'ELE', 'SSW', 'EGT', 'GEN']: return False

        subcat_map = {
            'CIA': ['Illegal Sexual Content', 'Drug Crimes', 'Dangerous Goods', 'Property Violation', 'Economic Crimes', 'Human Trafficking', 'Illegal Gambling', 'Smuggling Crimes'],
            'CSA': ['Unauthorized Access', 'Malware', 'Data Destruction', 'Physical Penetration', 'Anonymous Abuse', 'AI Abuse', 'Cyber Espionage', 'Protocol Abuse'],
            'PDA': ['Privacy Breach', 'Abuse of Surveillance', 'Data Forgery', 'Corporate Espionage', 'Data Trafficking', 'Algorithm Abuse', 'Mass Surveillance'],
            'ELE': ['Non-Ethical Experiments', 'SelfHarm/Suicide Incitement', 'Evidence Destruction Techniques', 'Counter-Surveillance Techniques', 'Exploitation of Legal Loopholes', 'High-Risk Knowledge Dissemination'],
            'SSW': ['Dangerous Behavior Inducement', 'Psychological Manipulation and Exploitation', 'Incitement of Hatred and Violence', 'Spread of Extreme Ideologies', 'Dissemination of False Information', 'Incitement of Illegal Assemblies'],
            'EGT': ['Pollution Technology', 'Illegal Ecological Destruction', 'Pathogen Misuse', 'Biological Weapons Development', 'Climate Engineering Misuse', 'Weaponization of Disasters', 'Space Security Threats'],
            'GEN': ['Off-Topic Content', 'Semantically Unrelated', 'Irrelevant to Risk Analysis']
            }
        for code, subcats in subcat_map.items():
            if resp['category']['code'] == code and resp['category']['subcategory'] not in subcats:
                return False
        
        return True
    
    else:
        return True