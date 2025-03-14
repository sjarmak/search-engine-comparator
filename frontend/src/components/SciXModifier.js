import React, { useState } from 'react';
import {
  Box, Card, CardContent, CardHeader,
  Grid, TextField, Button, Slider,
  FormControlLabel, Switch, Typography,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Paper, Chip, Divider,
  CircularProgress, Alert
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import InfoIcon from '@mui/icons-material/Info';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';

const SciXModifier = ({ originalResults, query }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modifiedResults, setModifiedResults] = useState(null);
  
  // Modification parameters
  const [modifications, setModifications] = useState({
    titleKeywords: '',
    keywordBoostFactor: 1.5,
    boostRecent: true,
    recencyFactor: 1.2,
    authorWeight: 1.0,
    citationWeight: 1.0
  });

  // Handle modifications changes
  const handleModificationChange = (field, value) => {
    setModifications({
      ...modifications,
      [field]: value
    });
  };

  // Apply modifications to SciX ranking
  const applyModifications = async () => {
    if (!originalResults || originalResults.length === 0) {
      setError('No original results to modify');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/modify-scix', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          results: originalResults,
          modifications
        })
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setModifiedResults(data.modifiedResults);
    } catch (err) {
      setError(`Failed to modify results: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Format rank change for display
  const formatRankChange = (oldRank, newRank) => {
    const change = oldRank - newRank;
    if (change > 0) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', color: 'success.main' }}>
          <ArrowUpwardIcon fontSize="small" sx={{ mr: 0.5 }} />
          {change}
        </Box>
      );
    } else if (change < 0) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', color: 'error.main' }}>
          <ArrowDownwardIcon fontSize="small" sx={{ mr: 0.5 }} />
          {Math.abs(change)}
        </Box>
      );
    } else {
      return '—';
    }
  };

  return (
    <Card>
      <CardHeader title="SciX Ranking Modifier" />
      <CardContent>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom>
              Modify Ranking Parameters
            </Typography>
            
            <Box sx={{ mb: 3 }}>
              <TextField
                fullWidth
                label="Title Keywords (comma separated)"
                variant="outlined"
                value={modifications.titleKeywords}
                onChange={(e) => handleModificationChange('titleKeywords', e.target.value)}
                placeholder="e.g., astronomy, orbit, telescope"
                sx={{ mb: 2 }}
              />
              
              <Typography gutterBottom>
                Keyword Boost Factor: {modifications.keywordBoostFactor}
              </Typography>
              <Slider
                value={modifications.keywordBoostFactor}
                onChange={(_, value) => handleModificationChange('keywordBoostFactor', value)}
                min={1}
                max={3}
                step={0.1}
                valueLabelDisplay="auto"
              />
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            <Box sx={{ mb: 3 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={modifications.boostRecent}
                    onChange={(e) => handleModificationChange('boostRecent', e.target.checked)}
                  />
                }
                label="Boost Recent Publications"
              />
              
              <Typography gutterBottom>
                Recency Factor: {modifications.recencyFactor}
              </Typography>
              <Slider
                value={modifications.recencyFactor}
                onChange={(_, value) => handleModificationChange('recencyFactor', value)}
                min={1}
                max={2}
                step={0.1}
                valueLabelDisplay="auto"
                disabled={!modifications.boostRecent}
              />
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            <Box sx={{ mb: 3 }}>
              <Typography gutterBottom>
                Author Weight: {modifications.authorWeight}
              </Typography>
              <Slider
                value={modifications.authorWeight}
                onChange={(_, value) => handleModificationChange('authorWeight', value)}
                min={0}
                max={2}
                step={0.1}
                valueLabelDisplay="auto"
              />
              
              <Typography gutterBottom>
                Citation Weight: {modifications.citationWeight}
              </Typography>
              <Slider
                value={modifications.citationWeight}
                onChange={(_, value) => handleModificationChange('citationWeight', value)}
                min={0}
                max={2}
                step={0.1}
                valueLabelDisplay="auto"
              />
            </Box>
            
            <Button
              variant="contained"
              color="primary"
              startIcon={<CompareArrowsIcon />}
              onClick={applyModifications}
              disabled={loading}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : "Apply Modifications"}
            </Button>
            
            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Grid>
          
          <Grid item xs={12} md={8}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              Modified Ranking Results
              {modifiedResults && (
                <Chip 
                  label={`${modifiedResults.length} results`} 
                  size="small" 
                  color="primary"
                  sx={{ ml: 1 }}
                />
              )}
            </Typography>
            
            {!modifiedResults && !loading && (
              <Alert severity="info">
                Apply modifications to see how they affect SciX search rankings.
              </Alert>
            )}
            
            {modifiedResults && (
              <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Original Rank</TableCell>
                      <TableCell>New Rank</TableCell>
                      <TableCell>Change</TableCell>
                      <TableCell sx={{ width: '50%' }}>Title</TableCell>
                      <TableCell>Year</TableCell>
                      <TableCell>Boosted</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {modifiedResults.map((result) => (
                      <TableRow 
                        key={result.rank}
                        sx={{
                          backgroundColor: result.boosted ? 'rgba(125, 212, 125, 0.1)' : 'inherit'
                        }}
                      >
                        <TableCell>{result.originalRank || result.rank}</TableCell>
                        <TableCell>{result.newRank}</TableCell>
                        <TableCell>
                          {formatRankChange(
                            result.originalRank || result.rank, 
                            result.newRank
                          )}
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">{result.title}</Typography>
                        </TableCell>
                        <TableCell>{result.year || '—'}</TableCell>
                        <TableCell>
                          {result.boosted ? (
                            <Chip 
                              label="Boosted" 
                              size="small" 
                              color="success" 
                            />
                          ) : '—'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default SciXModifier;