from src.polish import Polish, PolishStepUpdateCondition


class TestPolish:
    def test_reset_parameters(self):
        my_string = "my_non_sense_string"
        my_int = 999
        my_float = 0.999
        polish_1 = Polish()
        polish_1.user_best_bound = my_int
        polish_1.fix_only_non_zero = not polish_1.fix_only_non_zero
        polish_1.path = my_string
        polish_1.num_threads = my_int
        polish_1.prefix_log = my_string
        polish_1.prefix_sol = my_string
        polish_1.debug_mode = not polish_1.debug_mode
        polish_1.max_time_no_improve_in_iteration = my_int
        polish_1.max_total_time_no_improve = my_int
        polish_1.epsilon = my_int
        polish_1.max_perc_time_find_1_sol = my_float
        conditions = [PolishStepUpdateCondition(max_gap=0.4, perc_decrease=0.1)]
        polish_1.set_step_update_conditions(user_conditions=conditions)

        polish_1.reset_parameters()

        polish_2 = Polish()
        assert polish_1.user_best_bound == polish_2.user_best_bound
        assert polish_1.fix_only_non_zero == polish_2.fix_only_non_zero
        assert polish_1.path == polish_2.path
        assert polish_1.num_threads == polish_2.num_threads
        assert polish_1.prefix_log == polish_2.prefix_log
        assert polish_1.prefix_sol == polish_2.prefix_sol
        assert polish_1.debug_mode == polish_2.debug_mode
        assert (
            polish_1.max_time_no_improve_in_iteration
            == polish_2.max_time_no_improve_in_iteration
        )
        assert polish_1.max_total_time_no_improve == polish_2.max_total_time_no_improve
        assert polish_1.epsilon == polish_2.epsilon
        assert polish_1.max_perc_time_find_1_sol == polish_2.max_perc_time_find_1_sol
        assert polish_1.update_step_conditions == polish_2.update_step_conditions
