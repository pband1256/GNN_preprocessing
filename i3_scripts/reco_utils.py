from icecube import dataclasses, dataio, icetray, MuonGun
from icecube.icetray import I3Units
import icecube.MuonGun
import numpy as np
from icecube.weighting.weighting import from_simprod
from icecube.icetray import I3Units

nu_pdg = [12, 14, 16, -12, -14, -16]

def is_data(frame):
    if ('I3MCWeightDict' in frame) or ('CorsikaWeightMap' in frame) or ('MCPrimary' in frame) or ('I3MCTree' in frame):
        return False
    else:
        return True

def has_signature(p, surface):
    intersections = surface.intersection(p.pos, p.dir)
    if p.is_neutrino:
        return -1
    if not np.isfinite(intersections.first):
        return -1
    if p.is_cascade:
        if intersections.first <= 0 and intersections.second > 0:
            return 0  # starting event
        else:
            return -1  # no hits
    elif p.is_track:
        if intersections.first <= 0 and intersections.second > 0:
            return 0  # starting event
        elif intersections.first > 0 and intersections.second > 0:
            if p.length <= intersections.first:
                return -1  # no hit
            elif p.length > intersections.second:
                return 1  # through-going event
            else:
                return 2  # stopping event
        else:
            return -1

def find_all_neutrinos(p_frame):
    if is_data(p_frame):
        return True
    I3Tree = p_frame['I3MCTree']
    # find first neutrino as seed for find_particle
    for p in I3Tree.get_primaries():
        if p.pdg_encoding in nu_pdg:
            break
    all_nu = [i for i in crawl_neutrinos(p, I3Tree, plist=[]) if len(i) > 0]
    return all_nu[-1][0]

def crawl_neutrinos(p, I3Tree, level=0, plist = []):
    if len(plist) < level+1:
        plist.append([])
    if (p.is_neutrino) & np.isfinite(p.length):
        plist[level].append(p) 
    children = I3Tree.children(p)
    if len(children) < 10:
        for child in children:
            crawl_neutrinos(child, I3Tree, level=level+1, plist=plist)
    return plist

# Generation of the Classification Label
def classify(p_frame, gcdfile=None, surface=None):
    if is_data(p_frame):
        return True
    pclass = 101 # only for security
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(p_frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    I3Tree = p_frame['I3MCTree']
    neutrino = find_all_neutrinos(p_frame)
    children = I3Tree.children(neutrino)
    p_types = [np.abs(child.pdg_encoding) for child in children]
    p_strings = [child.type_string for child in children]
    p_frame.Put("visible_nu", neutrino)
    IC_hit = np.any([((has_signature(tp, surface) != -1) & np.isfinite(tp.length)) for tp in children])
    if p_frame['I3MCWeightDict']['InteractionType'] == 3 and (len(p_types) == 1 and p_strings[0] == 'Hadrons'):
        pclass = 7  # Glashow Cascade
    else:
        if (11 in p_types) or (p_frame['I3MCWeightDict']['InteractionType'] == 2):
            if IC_hit:
                pclass = 1  # Cascade
            else:
                pclass = 0 # Uncontainced Cascade
        elif (13 in p_types):
            mu_ind = p_types.index(13)
            p_frame.Put("visible_track", children[mu_ind])
            if not IC_hit:
                pclass = 11 # Passing Track
            elif p_frame['I3MCWeightDict']['InteractionType'] == 3:
                if has_signature(children[mu_ind], surface) == 0:
                    pclass = 8  # Glashow Track
            elif has_signature(children[mu_ind], surface) == 0:
                pclass = 3  # Starting Track
            elif has_signature(children[mu_ind], surface) == 1:
                pclass = 2  # Through Going Track
            elif has_signature(children[mu_ind], surface) == 2:
                pclass = 4  # Stopping Track
        elif (15 in p_types):
            tau_ind = p_types.index(15)
            p_frame.Put("visible_track", children[tau_ind])
            if not IC_hit:
                pclass = 12 # uncontained tau something...
            else:
                # consider to use the interactiontype here...
                if p_frame['I3MCWeightDict']['InteractionType'] == 3:
                    pclass =  9  # Glashow Tau
                else:
                    had_ind = p_strings.index('Hadrons')
                    try:
                        tau_child = I3Tree.children(children[tau_ind])[-1]
                    except:
                        tau_child = None
                    if tau_child:
                        if np.abs(tau_child.pdg_encoding) == 13:
                            if has_signature(tau_child, surface) == 0:
                                pclass = 3  # Starting Track
                            if has_signature(tau_child, surface) == 1:
                                pclass = 2  # Through Going Track
                            if has_signature(tau_child, surface) == 2:
                                pclass = 4  # Stopping Track
                        else:
                            if has_signature(children[tau_ind], surface) == 0 and has_signature(tau_child, surface) == 0:
                                pclass = 5  # Double Bang
                            if has_signature(children[tau_ind], surface) == 0 and has_signature(tau_child, surface) == -1:
                                pclass = 3  # Starting Track
                            if has_signature(children[tau_ind], surface) == 2 and has_signature(tau_child, surface) == 0:
                                pclass = 6  # Stopping Tau
                            if has_signature(children[tau_ind], surface) == 1:
                                pclass = 2  # Through Going Track
                    else: # Tau Decay Length to large, so no childs are simulated
                        if has_signature(children[tau_ind], surface) == 0:
                            pclass = 3 # Starting Track
                        if has_signature(children[tau_ind], surface) == 1:
                            pclass = 2  # Through Going Track
                        if has_signature(children[tau_ind], surface) == 2:
                            pclass = 4  # Stopping Track
        else:
            pclass = 100 # unclassified
    #print('Classification: {}'.format(pclass))
    p_frame.Put("classification", icetray.I3Int(pclass))
    return

def get_most_E_muon_info(frame, gcdfile=None, surface=None, tracklist='MMCTrackList', mctree='I3MCTree'):
    if is_data(frame):
        return True
    if tracklist not in frame:
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(frame['I3Geometry'], padding=0)
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    e0_list = []
    edep_list = []
    particle_list = []
    if isinstance(frame[tracklist], icecube.dataclasses.I3Particle):
        tlist = [frame[tracklist]]
    else:
        tlist=icecube.MuonGun.Track.harvest(frame[mctree], frame[tracklist])
    for track in tlist:
        # Find distance to entrance and exit from sampling volume
        intersections = surface.intersection(track.pos, track.dir)
        # Get the corresponding energies
        e0, e1 = track.get_energy(intersections.first), track.get_energy(intersections.second)
        e0_list.append(e0)
        particle_list.append(track)
        # Accumulate
        edep_list.append((e0-e1))
    edep_list = np.array(edep_list)
    inds = np.argsort(edep_list)[::-1]
    e0_list = np.array(e0_list)[inds]
    particle_list = np.array(particle_list)[inds]
    if len(particle_list) == 0:
        print('No clear muon')
    else:
        #frame.Put("Reconstructed_Muon", particle_list[0])
        frame.Put("mu_E_on_entry", dataclasses.I3Double(e0_list[0]))
        frame.Put("mu_E_deposited", dataclasses.I3Double(edep_list[inds][0]))
    return
