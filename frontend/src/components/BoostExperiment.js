import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, CardHeader, Grid, TextField, Button,
  Slider, FormControlLabel, Switch, Typography, FormControl,
  InputLabel, Select, MenuItem, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Paper, Chip, Divider,
  CircularProgress, Alert, Tooltip, IconButton, Collapse
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
  const [results, setResults] = useState(null);
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
  
  // Update a single boost config parameter
  const updateBoostConfig = (key, value) => {
    setBoostConfig({
      ...boostConfig,
      [key]: value
    });
  };
  
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
      console.log("Sending request to boost-experiment endpoint with config:", boostConfig);
      
      // Use the full API URL here instead of relative path
      const response = await fetch(`${API_URL}/api/boost-experiment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          results: originalResults,
          boostConfig
        })
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }
      
      // Process and parse the response carefully
      let data;
      try {
        const text = await response.text();
        console.log("Raw response text:", text);
        data = JSON.parse(text);
      } catch (jsonError) {
        throw new Error(`JSON parsing error: ${jsonError.message}`);
      }
      
      if (data.status === "error") {
        throw new Error(data.message);
      }
      
      // Process and normalize the results
      const processedResults = (data.results || []).map(result => {
        // Ensure all required fields exist with fallbacks
        return {
          ...result,
          newRank: result.rank,
          rankChange: result.rankChange || 0,
          citations: result.citations || result.citation_count || 0,
          year: result.year || '',
          // Ensure boost factors exist
          finalBoost: result.totalBoost || 0,
          // Add direct access to boost factors for tooltip display
          boostFactors: {
            ...result.boostFactors,
            citeBoost: result.citeBoost || result.boostFactors?.citeBoost || 0,
            recencyBoost: result.recencyBoost || result.boostFactors?.recencyBoost || 0,
            doctypeBoost: result.doctypeBoost || result.boostFactors?.doctypeBoost || 0,
            refereedBoost: result.refereedBoost || result.boostFactors?.refereedBoost || 0,
          }
        };
      });
      
      console.log("Processed results:", processedResults);
      setResults(processedResults);
      
      // Store debug info about the first result for debugging panel
      if (processedResults.length > 0) {
        const firstResult = processedResults[0];
        setDebugInfo({
          firstResult,
          citationFields: {
            citations: firstResult.citations,
            citation_count: firstResult.citation_count,
            citationCount: firstResult.citationCount,
          },
          boostFields: {
            boostFactors: firstResult.boostFactors,
            citeBoost: firstResult.citeBoost,
            recencyBoost: firstResult.recencyBoost,
            doctypeBoost: firstResult.doctypeBoost,
            refereedBoost: firstResult.refereedBoost,
            totalBoost: firstResult.totalBoost,
            finalBoost: firstResult.finalBoost,
          }
        });
      }
      
    } catch (err) {
      console.error("Error in boost experiment:", err);
      setError(`Failed to process boost experiment: ${err.message}`);
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
  
  // Run experiment initially and on config changes
  useEffect(() => {
    if (originalResults && originalResults.length > 0) {
      applyBoosts();
    }
  }, []);
  
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
  
  return (
    <Card>
      <CardHeader 
        title="Boost Factor Experiment" 
        subheader="Test how different boost factors affect search result ranking"
        action={
          <Button
            startIcon={<BugReportIcon />}
            variant="outlined"
            size="small"
            color="warning"
            onClick={() => setDebugMode(!debugMode)}
          >
            {debugMode ? 'Hide Debug' : 'Debug Mode'}
          </Button>
        }
      />
      <CardContent>
        <Grid container spacing={3}>
          {/* Configuration Panel */}
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Boost Configuration
                  <Tooltip title="Configure boost factors to see how they affect result ranking">
                    <IconButton size="small" sx={{ ml: 1 }}>
                      <HelpOutlineIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Typography>
                
                {/* Citation Boost */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Citation Boost
                    <FormControlLabel
                      control={
                        <Switch
                          checked={boostConfig.enableCiteBoost}
                          onChange={(e) => updateBoostConfig('enableCiteBoost', e.target.checked)}
                          size="small"
                        />
                      }
                      label="Enable"
                      labelPlacement="start"
                      sx={{ ml: 1 }}
                    />
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Weight factor for citation counts
                  </Typography>
                  
                  <Slider
                    disabled={!boostConfig.enableCiteBoost}
                    value={boostConfig.citeBoostWeight}
                    min={0}
                    max={2}
                    step={0.1}
                    valueLabelDisplay="auto"
                    onChange={(_, value) => updateBoostConfig('citeBoostWeight', value)}
                  />
                </Box>
                
                {/* Recency Boost */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Recency Boost
                    <FormControlLabel
                      control={
                        <Switch
                          checked={boostConfig.enableRecencyBoost}
                          onChange={(e) => updateBoostConfig('enableRecencyBoost', e.target.checked)}
                          size="small"
                        />
                      }
                      label="Enable"
                      labelPlacement="start"
                      sx={{ ml: 1 }}
                    />
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Weight factor for paper recency
                  </Typography>
                  
                  <Slider
                    disabled={!boostConfig.enableRecencyBoost}
                    value={boostConfig.recencyBoostWeight}
                    min={0}
                    max={2}
                    step={0.1}
                    valueLabelDisplay="auto"
                    onChange={(_, value) => updateBoostConfig('recencyBoostWeight', value)}
                  />
                  
                  <FormControl fullWidth sx={{ mt: 2 }} size="small">
                    <InputLabel>Recency Function</InputLabel>
                    <Select
                      value={boostConfig.recencyFunction}
                      label="Recency Function"
                      disabled={!boostConfig.enableRecencyBoost}
                      onChange={(e) => updateBoostConfig('recencyFunction', e.target.value)}
                    >
                      <MenuItem value="exponential">Exponential (e^-m*age)</MenuItem>
                      <MenuItem value="inverse">Inverse (1/1+m*age)</MenuItem>
                      <MenuItem value="linear">Linear (1-m*age)</MenuItem>
                      <MenuItem value="sigmoid">Sigmoid (1/(1+e^(m*(age-α))))</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }} gutterBottom>
                    Decay multiplier
                  </Typography>
                  
                  <Slider
                    disabled={!boostConfig.enableRecencyBoost}
                    value={boostConfig.recencyMultiplier}
                    min={0.01}
                    max={0.2}
                    step={0.01}
                    valueLabelDisplay="auto"
                    onChange={(_, value) => updateBoostConfig('recencyMultiplier', value)}
                  />
                </Box>
                
                {/* Document Type Boost */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Document Type Boost
                    <FormControlLabel
                      control={
                        <Switch
                          checked={boostConfig.enableDoctypeBoost}
                          onChange={(e) => updateBoostConfig('enableDoctypeBoost', e.target.checked)}
                          size="small"
                        />
                      }
                      label="Enable"
                      labelPlacement="start"
                      sx={{ ml: 1 }}
                    />
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Weight factor for document type
                  </Typography>
                  
                  <Slider
                    disabled={!boostConfig.enableDoctypeBoost}
                    value={boostConfig.doctypeBoostWeight}
                    min={0}
                    max={2}
                    step={0.1}
                    valueLabelDisplay="auto"
                    onChange={(_, value) => updateBoostConfig('doctypeBoostWeight', value)}
                  />
                </Box>
                
                {/* Refereed Boost */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Refereed Boost
                    <FormControlLabel
                      control={
                        <Switch
                          checked={boostConfig.enableRefereedBoost}
                          onChange={(e) => updateBoostConfig('enableRefereedBoost', e.target.checked)}
                          size="small"
                        />
                      }
                      label="Enable"
                      labelPlacement="start"
                      sx={{ ml: 1 }}
                    />
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Weight factor for refereed papers
                  </Typography>
                  
                  <Slider
                    disabled={!boostConfig.enableRefereedBoost}
                    value={boostConfig.refereedBoostWeight}
                    min={0}
                    max={2}
                    step={0.1}
                    valueLabelDisplay="auto"
                    onChange={(_, value) => updateBoostConfig('refereedBoostWeight', value)}
                  />
                </Box>
                
                {/* Combination Method */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Combination Method
                  </Typography>
                  
                  <FormControl fullWidth size="small">
                    <InputLabel>Combination Method</InputLabel>
                    <Select
                      value={boostConfig.combinationMethod}
                      label="Combination Method"
                      onChange={(e) => updateBoostConfig('combinationMethod', e.target.value)}
                    >
                      <MenuItem value="sum">Sum</MenuItem>
                      <MenuItem value="product">Product</MenuItem>
                      <MenuItem value="max">Maximum</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                  <Button 
                    variant="outlined" 
                    startIcon={<ReplayIcon />} 
                    onClick={resetDefaults}
                  >
                    Reset Defaults
                  </Button>
                  
                  <Button 
                    variant="contained" 
                    color="primary" 
                    onClick={applyBoosts}
                    disabled={loading}
                  >
                    Apply Boosts
                    {loading && (
                      <CircularProgress 
                        size={24} 
                        color="inherit" 
                        sx={{ ml: 1 }} 
                      />
                    )}
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Results Panel */}
          <Grid item xs={12} md={8}>
            <Typography variant="h6" gutterBottom>
              Ranking Results
              {loading && (
                <CircularProgress 
                  size={20} 
                  color="primary" 
                  sx={{ ml: 1 }} 
                />
              )}
            </Typography>
            
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}
            
            {!results && !loading && !error && (
              <Alert severity="info">
                Configure and apply boost factors to see how they affect the ranking.
              </Alert>
            )}
            
            {/* Debug Panel */}
            <Collapse in={debugMode}>
              {renderDebugPanel()}
            </Collapse>
            
            {results && results.length > 0 && (
              <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Original Rank</TableCell>
                      <TableCell>New Rank</TableCell>
                      <TableCell>Change</TableCell>
                      <TableCell>Title</TableCell>
                      <TableCell>Year</TableCell>
                      <TableCell>Citations</TableCell>
                      <TableCell>Boost Score</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {results.map((result, index) => (
                      <TableRow 
                        key={result.identifier || index}
                        sx={{
                          backgroundColor: result.rankChange > 5 ? 'rgba(76, 175, 80, 0.1)' : 
                                           result.rankChange < -5 ? 'rgba(244, 67, 54, 0.1)' : 
                                           'inherit'
                        }}
                      >
                        <TableCell>{result.originalRank}</TableCell>
                        <TableCell>{result.newRank}</TableCell>
                        <TableCell>
                          {formatRankChange(result.rankChange)}
                        </TableCell>
                        <TableCell>
                          <Tooltip title={
                            <Box>
                              <Typography variant="caption" display="block">
                                <strong>Cite Boost:</strong> {formatBoostFactor(result.boostFactors?.citeBoost)}
                              </Typography>
                              <Typography variant="caption" display="block">
                                <strong>Recency Boost:</strong> {formatBoostFactor(result.boostFactors?.recencyBoost)}
                              </Typography>
                              <Typography variant="caption" display="block">
                                <strong>Doctype Boost:</strong> {formatBoostFactor(result.boostFactors?.doctypeBoost)}
                              </Typography>
                              <Typography variant="caption" display="block">
                                <strong>Refereed Boost:</strong> {formatBoostFactor(result.boostFactors?.refereedBoost)}
                              </Typography>
                              <Typography variant="caption" display="block">
                                <strong>Final Boost:</strong> {formatBoostFactor(result.finalBoost)}
                              </Typography>
                            </Box>
                          }>
                            <Typography variant="body2">{result.title}</Typography>
                          </Tooltip>
                        </TableCell>
                        <TableCell>{result.year || '—'}</TableCell>
                        <TableCell>{result.citations || '0'}</TableCell>
                        <TableCell>{formatBoostFactor(result.finalBoost)}</TableCell>
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

export default BoostExperiment;