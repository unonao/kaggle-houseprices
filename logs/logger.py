"""
lightgbmの学習履歴を標準入出力ではなくlogger経由で出力

参考：https://amalog.hateblo.jp/entry/lightgbm-logging-callback
"""

import logging
from lightgbm.callback import _format_eval_result


def log_best(model, metric):
    logging.debug("\tbest_iteration: {}".format(model.best_iteration))
    logging.debug("\tbest_score: {}".format(model.best_score['valid_0'][metric]))


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([
                _format_eval_result(x, show_stdv)
                for x in env.evaluation_result_list
            ])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback


def log_evaluation_xgb(logger, period=1, show_stdv=True):
    """Create a callback that logs evaluation result with logger.

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
         Whether show stdv if provided

    Returns
    -------
    callback : function
        A callback that logs evaluation every period iterations into logger.
    """

    def _fmt_metric(value, show_stdv=True):
        """format metric string"""
        if len(value) == 2:
            return '%s:%g' % (value[0], value[1])
        elif len(value) == 3:
            if show_stdv:
                return '%s:%g+%g' % (value[0], value[1], value[2])
            else:
                return '%s:%g' % (value[0], value[1])
        else:
            raise ValueError("wrong metric value")

    def callback(env):
        if env.rank != 0 or len(env.evaluation_result_list) == 0 or period is False:
            return
        i = env.iteration
        if i % period == 0 or i + 1 == env.begin_iteration or i + 1 == env.end_iteration:
            msg = '\t'.join([_fmt_metric(x, show_stdv) for x in env.evaluation_result_list])
            logger.info('[%d]\t%s\n' % (i, msg))

    return callback
