def solve(img_ids, pair_scores_dict, BEAM_SIZE=10):
    num_frames=img_ids.shape[0]
    beam=[]                 # Contains top BEAM_SIZE tuples (score,solution_list),solution_list is a set
    all_solutions=[]        # Contains all tuples (score,solution_list) at a particular step 
    for step_number in range(num_frames):
        if step_number==0:
            for i in range(num_frames):
                all_solutions.append((0,{i},[i])) # score, solution, list
            beam=sorted(all_solutions, key=lambda tup: tup[0], reverse=True)
            beam=beam[:]
            all_solutions=[]
        else:
            all_solutions=[]
            for solution in beam:
                curr_list=solution[2]
                curr_solution=solution[1]
                curr_score=solution[0]
                last_ele=curr_list[-1]
                for i in range(num_frames):
                    if i not in curr_solution:
                        pair_score=pair_scores_dict[(last_ele,i)]
                        temp_sol=set(curr_solution)
                        temp_sol.add(i)
                        temp_list=curr_list[:]
                        temp_list.append(i)
                        all_solutions.append((curr_score+pair_score,temp_sol,temp_list))
            beam=sorted(all_solutions, key=lambda tup: tup[0], reverse=True)
            beam=beam[:BEAM_SIZE]

    solution_indices=beam[0][2]
    return img_ids[solution_indices] # GOLD REFERENCE 0 -> len(img_ids)


def solve_1(test_paragraph,pair_scores_dict,BEAM_SIZE=64):
    
    num_sentences=test_paragraph.shape[0]
    beam=[]                 # Contains top BEAM_SIZE tuples (score,solution_list),solution_list is a set
    all_solutions=[]        # Contains all tuples (score,solution_list) at a particular step 
    for step_number in range(num_sentences):
        if step_number==0:
            for i in range(num_sentences):
                all_solutions.append((0,{i},[i]))
            beam=sorted(all_solutions, key=lambda tup: tup[0],reverse=True)
            beam=beam[:BEAM_SIZE]
            all_solutions=[]
        else:
            all_solutions=[]
            for solution in beam:
                curr_list=solution[2]
                curr_solution=solution[1]
                curr_score=solution[0]
                last_ele=curr_list[-1]
                for i in range(num_sentences):
                    if i not in curr_solution:
                        pair_score=pair_scores_dict[(last_ele,i)][0].item()
                        temp_sol=set(curr_solution)
                        temp_sol.add(i)
                        temp_list=curr_list[:]
                        temp_list.append(i)
                        all_solutions.append((curr_score+pair_score,temp_sol,temp_list))
            beam=sorted(all_solutions, key=lambda tup: tup[0],reverse=True)
            beam=beam[:BEAM_SIZE]

    solution_indices=beam[0][2]
    return test_paragraph[solution_indices]