import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, CardHeader, Grid, TextField, Button,
  Slider, FormControlLabel, Switch, Typography, FormControl,
  InputLabel, Select, MenuItem, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Paper, Chip, Divider,
  CircularProgress, Alert, Tooltip, IconButton, Collapse, List, ListItem, ListItemText
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import ReplayIcon from '@mui/icons-material/Replay';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import BugReportIcon from '@mui/icons-material/BugReport';

// Get API URL from environment or use default
const API_URL = process.env.REACT_APP_API_URL || 'https://search-engine-comparator-api.onrender.com';

/**
 * Component for experimenting with different boost factors and their impact on ranking
 * 
 * @param {Object} props - Component props
 * @param {Array} props.originalResults - The original search results to re-rank
 * @param {string} props.query - The search query used to retrieve results
 */
const BoostExperiment = ({ originalResults, query }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [boostedResults, setBoostedResults] = useState(null);
  const [debugMode, setDebugMode] = useState(false);
  const [debugInfo, setDebugInfo] = useState(null);
  
  // Boost configuration state
  const [boostConfig, setBoostConfig] = useState({
    // Citation boost
    enableCiteBoost: true,
    citeBoostWeight: 1.0,
    
    // Recency boost
    enableRecencyBoost: true,
    recencyBoostWeight: 1.0,
    recencyFunction: "exponential", // Changed to match backend default
    recencyMultiplier: 0.01, // Changed to match backend default
    recencyMidpoint: 36,
    
    // Document type boost
    enableDoctypeBoost: true,
    doctypeBoostWeight: 1.0,
    
    // Refereed boost
    enableRefereedBoost: true,
    refereedBoostWeight: 1.0,
    
    // Combination method
    combinationMethod: "sum"
  });
  
  // Debug logging
  useEffect(() => {
    console.log('BoostExperiment mounted with:', {
      originalResultsLength: originalResults?.length,
      query,
      boostConfig
    });
    
    if (originalResults?.length > 0) {
      console.log("Sample result metadata:", {
        citation: originalResults[0].citation_count,
        year: originalResults[0].year,
        doctype: originalResults[0].doctype,
        properties: originalResults[0].property
      });
    }
  }, [originalResults, query, boostConfig]);
  
  // Apply the boost experiment
  const applyBoosts = async () => {
    if (!originalResults || originalResults.length === 0) {
      setError('No original results to modify');
      return;
    }
    
    setLoading(true);
    setError(null);
    setDebugInfo(null);
    
    try {
      console.log('Sending boost experiment request with:', {
        query,
        results: originalResults,
        boostConfig
      });
      
      const response = await fetch(`${API_URL}/api/boost-experiment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          results: originalResults,
          boostConfig
        })
      });
      
      console.log('Boost experiment response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Boost experiment error:', errorText);
        throw new Error(`Failed to apply boosts: ${errorText}`);
      }
      
      const data = await response.json();
      console.log('Received boosted results:', data);
      setBoostedResults(data);
      
      // Store debug info about the first result for debugging panel
      if (data.results && data.results.length > 0) {
        const firstResult = data.results[0];
        setDebugInfo({
          firstResult,
          citationFields: {
            citations: firstResult.citations,
            citation_count: firstResult.citation_count,
            citationCount: firstResult.citationCount,
          },
          boostFields: {
            boostFactors: firstResult.boostFactors,
            citeBoost: firstResult.citeBoost || firstResult.boostFactors?.citeBoost || 0,
            recencyBoost: firstResult.recencyBoost || firstResult.boostFactors?.recencyBoost || 0,
            doctypeBoost: firstResult.doctypeBoost || firstResult.boostFactors?.doctypeBoost || 0,
            refereedBoost: firstResult.refereedBoost || firstResult.boostFactors?.refereedBoost || 0,
            totalBoost: firstResult.totalBoost,
            finalBoost: firstResult.finalBoost,
          }
        });
      }
      
    } catch (err) {
      console.error('Error in boost experiment:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Format rank change display
  const formatRankChange = (change) => {
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
  
  // Format boost factor display
  const formatBoostFactor = (value) => {
    if (value === undefined || value === null) return '—';
    return value.toFixed(2);
  };
  
  // Reset to default boost configuration
  const resetDefaults = () => {
    setBoostConfig({
      enableCiteBoost: true,
      citeBoostWeight: 1.0,
      enableRecencyBoost: true,
      recencyBoostWeight: 1.0,
      recencyFunction: "exponential", // Changed to match backend default
      recencyMultiplier: 0.01, // Changed to match backend default
      recencyMidpoint: 36,
      enableDoctypeBoost: true,
      doctypeBoostWeight: 1.0,
      enableRefereedBoost: true,
      refereedBoostWeight: 1.0,
      combinationMethod: "sum"
    });
  };
  
  // Apply boosts whenever configuration changes
  useEffect(() => {
    if (originalResults && originalResults.length > 0) {
      applyBoosts();
    } else {
      console.log('No original results to boost');
    }
  }, [boostConfig, originalResults]);
  
  // Debug component to inspect fields and values
  const renderDebugPanel = () => {
    if (!debugInfo) return null;
    
    return (
      <Box sx={{ mt: 2, mb: 2, border: 1, borderColor: 'warning.light', p: 2, borderRadius: 1 }}>
        <Typography variant="h6" color="warning.main" gutterBottom>
          Debug Information
        </Typography>
        
        <Typography variant="subtitle2" gutterBottom>Citation Fields</Typography>
        <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Field Name</TableCell>
                <TableCell>Present</TableCell>
                <TableCell>Value</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(debugInfo.citationFields).map(([field, value]) => (
                <TableRow key={field}>
                  <TableCell>{field}</TableCell>
                  <TableCell>
                    {value !== undefined && value !== null ? (
                      <Chip label="Yes" size="small" color="success" />
                    ) : (
                      <Chip label="No" size="small" color="error" />
                    )}
                  </TableCell>
                  <TableCell>{value !== undefined && value !== null ? String(value) : 'N/A'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <Typography variant="subtitle2" gutterBottom>Boost Fields</Typography>
        <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Field Name</TableCell>
                <TableCell>Present</TableCell>
                <TableCell>Value</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(debugInfo.boostFields).map(([field, value]) => (
                <TableRow key={field}>
                  <TableCell>{field}</TableCell>
                  <TableCell>
                    {value !== undefined && value !== null ? (
                      <Chip label="Yes" size="small" color="success" />
                    ) : (
                      <Chip label="No" size="small" color="error" />
                    )}
                  </TableCell>
                  <TableCell>
                    {field === 'boostFactors' && typeof value === 'object' ? (
                      <Typography variant="caption" component="pre" sx={{ maxHeight: 100, overflow: 'auto' }}>
                        {JSON.stringify(value, null, 2)}
                      </Typography>
                    ) : (
                      value !== undefined && value !== null ? String(value) : 'N/A'
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <Typography variant="subtitle2" gutterBottom>First Result Raw Data</Typography>
        <Box 
          component="pre" 
          sx={{ 
            maxHeight: 200, 
            overflow: 'auto', 
            fontSize: '0.75rem', 
            bgcolor: 'grey.100', 
            p: 1, 
            borderRadius: 1 
          }}
        >
          {JSON.stringify(debugInfo.firstResult, null, 2)}
        </Box>
      </Box>
    );
  };
  
  if (!originalResults || originalResults.length === 0) {
    return (
      <Alert severity="warning">
        No results available for boost experiment. Please perform a search first.
      </Alert>
    );
  }
  
  const handleConfigChange = (field, value) => {
    setBoostConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  const renderBoostControls = () => (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">Boost Configuration</Typography>
        <Button
          startIcon={<ReplayIcon />}
          variant="outlined"
          size="small"
          onClick={resetDefaults}
        >
          Reset Defaults
        </Button>
      </Box>
      <Grid container spacing={2}>
        {/* Citation Boost */}
        <Grid item xs={12}>
          <Box sx={{ mb: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={boostConfig.enableCiteBoost}
                  onChange={(e) => handleConfigChange('enableCiteBoost', e.target.checked)}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography>Citation Boost</Typography>
                  <Tooltip title="Boost based on number of citations">
                    <IconButton size="small">
                      <HelpOutlineIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              }
            />
            {boostConfig.enableCiteBoost && (
              <Box sx={{ px: 2, mt: 1 }}>
                <Typography variant="caption" gutterBottom>Weight</Typography>
                <Slider
                  value={boostConfig.citeBoostWeight}
                  onChange={(_, value) => handleConfigChange('citeBoostWeight', value)}
                  min={0}
                  max={2}
                  step={0.1}
                  valueLabelDisplay="auto"
                  aria-label="Citation Boost Weight"
                />
              </Box>
            )}
          </Box>
        </Grid>

        {/* Recency Boost */}
        <Grid item xs={12}>
          <Box sx={{ mb: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={boostConfig.enableRecencyBoost}
                  onChange={(e) => handleConfigChange('enableRecencyBoost', e.target.checked)}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography>Recency Boost</Typography>
                  <Tooltip title="Boost based on publication year">
                    <IconButton size="small">
                      <HelpOutlineIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              }
            />
            {boostConfig.enableRecencyBoost && (
              <Box sx={{ px: 2, mt: 1 }}>
                <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                  <InputLabel>Decay Function</InputLabel>
                  <Select
                    value={boostConfig.recencyFunction}
                    onChange={(e) => handleConfigChange('recencyFunction', e.target.value)}
                    label="Decay Function"
                  >
                    <MenuItem value="exponential">Exponential</MenuItem>
                    <MenuItem value="inverse">Inverse</MenuItem>
                    <MenuItem value="linear">Linear</MenuItem>
                    <MenuItem value="sigmoid">Sigmoid</MenuItem>
                  </Select>
                </FormControl>
                <Typography variant="caption" gutterBottom>Weight</Typography>
                <Slider
                  value={boostConfig.recencyBoostWeight}
                  onChange={(_, value) => handleConfigChange('recencyBoostWeight', value)}
                  min={0}
                  max={2}
                  step={0.1}
                  valueLabelDisplay="auto"
                  aria-label="Recency Boost Weight"
                />
              </Box>
            )}
          </Box>
        </Grid>

        {/* Document Type Boost */}
        <Grid item xs={12}>
          <Box sx={{ mb: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={boostConfig.enableDoctypeBoost}
                  onChange={(e) => handleConfigChange('enableDoctypeBoost', e.target.checked)}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography>Document Type Boost</Typography>
                  <Tooltip title="Boost based on document type (article, review, etc.)">
                    <IconButton size="small">
                      <HelpOutlineIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              }
            />
            {boostConfig.enableDoctypeBoost && (
              <Box sx={{ px: 2, mt: 1 }}>
                <Typography variant="caption" gutterBottom>Weight</Typography>
                <Slider
                  value={boostConfig.doctypeBoostWeight}
                  onChange={(_, value) => handleConfigChange('doctypeBoostWeight', value)}
                  min={0}
                  max={2}
                  step={0.1}
                  valueLabelDisplay="auto"
                  aria-label="Document Type Boost Weight"
                />
              </Box>
            )}
          </Box>
        </Grid>

        {/* Refereed Boost */}
        <Grid item xs={12}>
          <Box sx={{ mb: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={boostConfig.enableRefereedBoost}
                  onChange={(e) => handleConfigChange('enableRefereedBoost', e.target.checked)}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography>Refereed Boost</Typography>
                  <Tooltip title="Boost for peer-reviewed papers">
                    <IconButton size="small">
                      <HelpOutlineIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              }
            />
            {boostConfig.enableRefereedBoost && (
              <Box sx={{ px: 2, mt: 1 }}>
                <Typography variant="caption" gutterBottom>Weight</Typography>
                <Slider
                  value={boostConfig.refereedBoostWeight}
                  onChange={(_, value) => handleConfigChange('refereedBoostWeight', value)}
                  min={0}
                  max={2}
                  step={0.1}
                  valueLabelDisplay="auto"
                  aria-label="Refereed Boost Weight"
                />
              </Box>
            )}
          </Box>
        </Grid>

        {/* Combination Method */}
        <Grid item xs={12}>
          <FormControl fullWidth size="small">
            <InputLabel>Combination Method</InputLabel>
            <Select
              value={boostConfig.combinationMethod}
              onChange={(e) => handleConfigChange('combinationMethod', e.target.value)}
              label="Combination Method"
            >
              <MenuItem value="sum">Sum</MenuItem>
              <MenuItem value="product">Product</MenuItem>
              <MenuItem value="max">Maximum</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>
    </Paper>
  );
  
  return (
    <Box sx={{ width: '100%', p: 2 }}>
      {!originalResults || originalResults.length === 0 ? (
        <Alert severity="warning">
          No results available for boost experiment. Please perform a search first.
        </Alert>
      ) : (
        <Grid container spacing={2}>
          {/* Boost Controls */}
          <Grid item xs={12} md={4}>
            {renderBoostControls()}
          </Grid>

          {/* Results Panel */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  Ranking Results
                </Typography>
                {loading && <CircularProgress size={24} sx={{ ml: 2 }} />}
                <Button
                  startIcon={<BugReportIcon />}
                  variant="outlined"
                  size="small"
                  color="warning"
                  onClick={() => setDebugMode(!debugMode)}
                  sx={{ ml: 2 }}
                >
                  {debugMode ? 'Hide Debug' : 'Debug Mode'}
                </Button>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}

              <Collapse in={debugMode}>
                {renderDebugPanel()}
              </Collapse>

              {boostedResults?.results ? (
                <TableContainer sx={{ maxHeight: 'calc(100vh - 300px)' }}>
                  <Table stickyHeader size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Rank</TableCell>
                        <TableCell>Change</TableCell>
                        <TableCell>Title</TableCell>
                        <TableCell align="right">Year</TableCell>
                        <TableCell align="right">Citations</TableCell>
                        <TableCell align="right">Boost</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {boostedResults.results.map((result) => (
                        <TableRow
                          key={result.bibcode || result.title}
                          sx={{
                            backgroundColor: result.rankChange > 5 ? 'success.light' :
                                           result.rankChange < -5 ? 'error.light' :
                                           'inherit'
                          }}
                        >
                          <TableCell>{result.rank}</TableCell>
                          <TableCell>{formatRankChange(result.rankChange)}</TableCell>
                          <TableCell>
                            <Tooltip title={
                              <Box>
                                <Typography variant="caption" display="block">
                                  Citations: {result.citation_count || 0}
                                </Typography>
                                <Typography variant="caption" display="block">
                                  Type: {result.doctype || 'Unknown'}
                                </Typography>
                                <Typography variant="caption" display="block">
                                  {result.property?.includes('REFEREED') ? 'Refereed' : 'Not Refereed'}
                                </Typography>
                                <Divider sx={{ my: 0.5 }} />
                                <Typography variant="caption" display="block">
                                  <strong>Boost Factors:</strong>
                                </Typography>
                                <Typography variant="caption" display="block">
                                  Citation: {formatBoostFactor(result.boostFactors?.citeBoost)}
                                </Typography>
                                <Typography variant="caption" display="block">
                                  Recency: {formatBoostFactor(result.boostFactors?.recencyBoost)}
                                </Typography>
                                <Typography variant="caption" display="block">
                                  Document Type: {formatBoostFactor(result.boostFactors?.doctypeBoost)}
                                </Typography>
                                <Typography variant="caption" display="block">
                                  Refereed: {formatBoostFactor(result.boostFactors?.refereedBoost)}
                                </Typography>
                              </Box>
                            }>
                              <Typography variant="body2">{result.title}</Typography>
                            </Tooltip>
                          </TableCell>
                          <TableCell align="right">{result.year || '—'}</TableCell>
                          <TableCell align="right">{result.citation_count || 0}</TableCell>
                          <TableCell align="right">{formatBoostFactor(result.finalBoost)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Alert severity="info">
                  Configure and apply boost factors to see how they affect the ranking.
                </Alert>
              )}
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default BoostExperiment;