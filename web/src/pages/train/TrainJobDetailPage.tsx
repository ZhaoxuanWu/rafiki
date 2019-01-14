/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';
import { Typography, Paper, CircularProgress,
  Table, TableHead, TableCell, TableBody, TableRow, Icon, IconButton } from '@material-ui/core';
import { Pageview } from '@material-ui/icons';
import * as moment from 'moment';

import { AppUtils } from '../../App';
import { AppRoute } from '../../app/AppNavigator';
import { Trial } from '../../../client/RafikiClient';

interface Props {
  classes: { [s: string]: any };
  appUtils: AppUtils;
  app: string;
  appVersion: number;
}

class TrainJobDetailPage extends React.Component<Props> {
  state: {
    trials: Trial[] | null
  } = {
    trials: null
  }

  async componentDidMount() {
    const { appUtils: { rafikiClient, showError }, app, appVersion } = this.props;
    const user = rafikiClient.getCurrentUser();
    try {
      const trials = await rafikiClient.getTrialsOfTrainJob(app, appVersion);
      this.setState({ trials });
    } catch (error) {
      showError(error, 'Failed to retrieve trials for train job');
    }
  }

  renderTrials() {
    const { appUtils: { appNavigator }, classes } = this.props;
    const { trials } = this.state;

    return (
      <Paper className={classes.trialsPaper}>
        <Table padding="dense">
          <TableHead>
            <TableRow>
              <TableCell padding="none"></TableCell>
              <TableCell>ID</TableCell>
              <TableCell>Model</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Score</TableCell>
              <TableCell>Started At</TableCell>
              <TableCell>Stopped At</TableCell>
              <TableCell>Duration</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {trials.map(x => {
              return (
                <TableRow key={x.id} hover>
                  <TableCell padding="none">
                    <IconButton onClick={() => {
                      const link = AppRoute.TRIAL_DETAIL
                        .replace(':trialId', x.id)
                      appNavigator.goTo(link);
                    }}>
                      <Pageview /> 
                    </IconButton>
                  </TableCell>
                  <TableCell>{x.id}</TableCell>
                  <TableCell>{x.model_name}</TableCell>
                  <TableCell>{x.status}</TableCell>
                  <TableCell>{x.score}</TableCell>
                  <TableCell>{moment(x.datetime_started).fromNow()}</TableCell>
                  <TableCell>{x.datetime_stopped ? moment(x.datetime_stopped).fromNow(): '-'}</TableCell>
                  <TableCell>{
                    x.datetime_stopped ? 
                      // @ts-ignore
                      moment.duration(x.datetime_stopped - x.datetime_started).humanize()
                        : '-'
                    }</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Paper>
    )
  }

  render() {
    const { classes, app, appVersion } = this.props;
    const { trials } = this.state;

    return (
      <React.Fragment>
        <Typography gutterBottom variant="h2">
          Train Job 
          <span className={classes.headerSub}>{`(${app} V${appVersion})`}</span>
        </Typography>
        <Typography gutterBottom variant="h3">Trials</Typography>
        {
          trials &&
          this.renderTrials()
        }
        {
          !trials &&
          <CircularProgress />
        }
      </React.Fragment>
    );
  }
}

const styles: StyleRulesCallback = (theme) => ({
  headerSub: {
    fontSize: theme.typography.h4.fontSize,
    margin: theme.spacing.unit * 2
  },
  trialsPaper: {
    overflowX: 'auto'
  }
});

export default withStyles(styles)(TrainJobDetailPage);