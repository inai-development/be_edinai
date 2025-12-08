/**
 * Superadministration Portal Dashboard
 * Main dashboard for superadministration portal
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  LogOut,
  Menu,
  X,
  Users,
  Settings,
  UploadCloud,
  FileSpreadsheet,
  BarChart3,
  Shield,
  AlertCircle,
  Home,
  FileText,
} from 'lucide-react';
import toast from 'react-hot-toast';
import Button from '../../components/ui/Button';

const SuperadministrationDashboard = () => {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(true);
  const [portalStatus, setPortalStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadForm, setUploadForm] = useState({
    std: '',
    subject: '',
    sem: '',
    board: '',
    chapterNumber: '',
    chapterTitle: '',
  });
  const [uploadFile, setUploadFile] = useState(null);
  const [lastUploadResult, setLastUploadResult] = useState(null);

  const uploadGuidelines = [
    {
      title: 'Single upload, multi-delivery',
      description: 'One PDF instantly appears inside every admin’s chapter suggestions list.',
    },
    {
      title: 'Preserve naming clarity',
      description: 'Use meaningful chapter numbers and titles so admins instantly recognize the content.',
    },
    {
      title: 'Supported formats',
      description: 'Only PDF files up to 15 MB are supported for now. Larger files should be compressed first.',
    },
  ];

  useEffect(() => {
    checkPortalStatus();
  }, []);

  const checkPortalStatus = async () => {
    try {
      const token = localStorage.getItem('superadmin_token');
      if (!token) {
        navigate('/superadministration/login');
        return;
      }

      const response = await fetch('/api/superadministration/portal/status', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      const data = await response.json();
      if (data.status) {
        setPortalStatus(data.data);
      } else {
        toast.error('Failed to load portal status');
      }
    } catch (error) {
      console.error('Portal status error:', error);
      toast.error('Failed to load portal status');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      const token = localStorage.getItem('superadmin_token');
      if (token) {
        await fetch('/api/superadministration/portal/logout', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      localStorage.removeItem('superadmin_token');
      localStorage.removeItem('superadmin_user');
      toast.success('Logged out successfully');
      navigate('/superadministration/login');
    }
  };

  const handleUploadFormChange = (field) => (event) => {
    const value = event.target.value;
    setUploadForm((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] || null;
    setUploadFile(file);
  };

  const handleGlobalUpload = async (event) => {
    event.preventDefault();

    if (!uploadForm.std || !uploadForm.subject || !uploadForm.chapterNumber) {
      toast.error('Standard, Subject, and Chapter Number are required.');
      return;
    }

    if (!uploadFile) {
      toast.error('Please select a PDF file.');
      return;
    }

    const token = localStorage.getItem('superadmin_token');
    if (!token) {
      toast.error('Session expired. Please log in again.');
      navigate('/superadministration/login');
      return;
    }

    try {
      setUploading(true);
      const formData = new FormData();
      formData.append('std', uploadForm.std);
      formData.append('subject', uploadForm.subject);
      formData.append('chapter_number', uploadForm.chapterNumber);
      formData.append('sem', uploadForm.sem);
      formData.append('board', uploadForm.board);
      if (uploadForm.chapterTitle) {
        formData.append('chapter_title', uploadForm.chapterTitle);
      }
      formData.append('pdf_file', uploadFile);

      const response = await fetch('/api/superadministration/portal/upload-chapter', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();
      if (!data.status) {
        throw new Error(data.message || 'Upload failed');
      }

      toast.success('Chapter PDF uploaded for all admins!');
      setLastUploadResult({
        createdRecords: data?.data?.created_records ?? 0,
        failures: data?.data?.failures ?? [],
      });
      setUploadForm({
        std: '',
        subject: '',
        sem: '',
        board: '',
        chapterNumber: '',
        chapterTitle: '',
      });
      setUploadFile(null);
      event.target.reset();
    } catch (error) {
      console.error('Global upload error:', error);
      toast.error(error.message || 'Failed to upload PDF');
    } finally {
      setUploading(false);
    }
  };

  const menuItems = [
    {
      icon: Home,
      label: 'Dashboard',
      onClick: () => navigate('/superadministration/dashboard'),
    },
  ];

  const dashboardCards = [
    {
      title: 'Total Users',
      value: '2,547',
      icon: Users,
      color: 'from-blue-500 to-blue-600',
    },
    {
      title: 'Active Sessions',
      value: '342',
      icon: Shield,
      color: 'from-green-500 to-green-600',
    },
    {
      title: 'System Health',
      value: '99.8%',
      icon: BarChart3,
      color: 'from-purple-500 to-purple-600',
    },
    {
      title: 'Pending Actions',
      value: '12',
      icon: AlertCircle,
      color: 'from-orange-500 to-orange-600',
    },
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto mb-4"></div>
          <p className="text-dark-text-secondary">Loading portal...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-dark-bg">
      {/* Sidebar */}
      <motion.div
        initial={{ x: -250 }}
        animate={{ x: sidebarOpen ? 0 : -250 }}
        transition={{ duration: 0.3 }}
        className="fixed left-0 top-0 h-screen w-64 bg-dark-bg-secondary border-r border-dark-border z-40"
      >
        <div className="p-6 border-b border-dark-border">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-dark-text-primary">
                Superadmin
              </h1>
              <p className="text-xs text-dark-text-secondary">Portal v2.0</p>
            </div>
          </div>
        </div>

        <nav className="p-4 space-y-2">
          {menuItems.map((item, index) => (
            <motion.button
              key={index}
              whileHover={{ x: 4 }}
              onClick={item.onClick}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-dark-text-secondary hover:text-dark-text-primary hover:bg-dark-bg transition-colors"
            >
              <item.icon className="w-5 h-5" />
              <span>{item.label}</span>
            </motion.button>
          ))}
        </nav>

        <div className="absolute bottom-4 left-4 right-4">
          <Button
            variant="secondary"
            size="medium"
            className="w-full flex items-center justify-center gap-2"
            onClick={handleLogout}
          >
            <LogOut className="w-4 h-4" />
            Logout
          </Button>
        </div>
      </motion.div>

      {/* Main Content */}
      <div className={`flex-1 ${sidebarOpen ? 'ml-64' : 'ml-0'} transition-all duration-300`}>
        {/* Top Bar */}
        <div className="bg-dark-bg-secondary border-b border-dark-border p-4 flex items-center justify-between">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-dark-bg rounded-lg transition-colors"
          >
            {sidebarOpen ? (
              <X className="w-6 h-6 text-dark-text-primary" />
            ) : (
              <Menu className="w-6 h-6 text-dark-text-primary" />
            )}
          </button>
          <h2 className="text-xl font-bold text-dark-text-primary">
            Superadministration Dashboard
          </h2>
          <div className="w-10"></div>
        </div>

        {/* Dashboard Content */}
        <div className="p-6 overflow-auto">
          {/* Status Alert */}
          {portalStatus && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 bg-green-500/10 border border-green-500/30 rounded-lg flex items-center gap-3"
            >
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <p className="text-green-400 text-sm">
                {portalStatus.portal_name} is operational
              </p>
            </motion.div>
          )}

          {/* Dashboard Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {dashboardCards.map((card, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="card p-6 hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-dark-text-secondary text-sm mb-1">
                      {card.title}
                    </p>
                    <p className="text-3xl font-bold text-dark-text-primary">
                      {card.value}
                    </p>
                  </div>
                  <div className={`bg-gradient-to-br ${card.color} p-3 rounded-lg`}>
                    <card.icon className="w-6 h-6 text-white" />
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="card p-6"
          >
            <h3 className="text-lg font-bold text-dark-text-primary mb-4">
              Quick Actions
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <Button
                variant="secondary"
                size="medium"
                className="w-full"
                onClick={() => navigate('/superadministration/users')}
              >
                Manage Users
              </Button>
              <Button
                variant="secondary"
                size="medium"
                className="w-full"
                onClick={() => navigate('/superadministration/security')}
              >
                Security Settings
              </Button>
              <Button
                variant="secondary"
                size="medium"
                className="w-full"
                onClick={() => navigate('/superadministration/analytics')}
              >
                View Analytics
              </Button>
            </div>
          </motion.div>

          {/* Global Chapter Upload */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-8"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-3 rounded-xl shadow-inner">
                <UploadCloud className="w-6 h-6 text-white" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-dark-text-secondary">
                  Superadmin exclusive
                </p>
                <h3 className="text-2xl font-bold text-dark-text-primary">
                  Distribute a chapter PDF globally
                </h3>
                <p className="text-dark-text-secondary text-sm">
                  One upload updates every admin’s chapter suggestions instantly.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[2fr_1fr] gap-6">
              <div className="rounded-2xl bg-gradient-to-br from-dark-bg-secondary to-dark-bg border border-dark-border/60 p-6 shadow-[0_20px_45px_rgba(15,15,30,0.35)]">
                <form className="space-y-6" onSubmit={handleGlobalUpload}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <FormField
                      label="Standard"
                      hint="e.g., Class 8"
                      required
                      value={uploadForm.std}
                      onChange={handleUploadFormChange('std')}
                    />
                    <FormField
                      label="Subject"
                      hint="e.g., Science"
                      required
                      value={uploadForm.subject}
                      onChange={handleUploadFormChange('subject')}
                    />
                    <FormField
                      label="Chapter Number"
                      hint="e.g., Chapter 5"
                      required
                      value={uploadForm.chapterNumber}
                      onChange={handleUploadFormChange('chapterNumber')}
                    />
                    <FormField
                      label="Chapter Title"
                      hint="Optional display title"
                      value={uploadForm.chapterTitle}
                      onChange={handleUploadFormChange('chapterTitle')}
                    />
                    <FormField
                      label="Semester"
                      hint="Optional"
                      value={uploadForm.sem}
                      onChange={handleUploadFormChange('sem')}
                    />
                    <FormField
                      label="Board"
                      hint="Optional"
                      value={uploadForm.board}
                      onChange={handleUploadFormChange('board')}
                    />
                  </div>

                  <div>
                    <label className="text-sm text-dark-text-secondary mb-2 block">Chapter PDF *</label>
                    <label className="relative flex flex-col items-center justify-center w-full min-h-[9rem] border border-dashed border-dark-border rounded-2xl cursor-pointer hover:border-purple-500/70 transition-all bg-dark-bg/40 backdrop-blur">
                      <input
                        type="file"
                        accept="application/pdf"
                        className="hidden"
                        onChange={handleFileChange}
                      />
                      <UploadCloud className="w-8 h-8 text-purple-400 mb-2" />
                      <span className="text-dark-text-secondary text-sm">
                        {uploadFile ? uploadFile.name : 'Click or drag PDF here'}
                      </span>
                      <span className="text-[11px] uppercase tracking-[0.3em] text-dark-text-muted mt-2">
                        PDF up to 15MB
                      </span>
                    </label>
                  </div>

                  <Button
                    type="submit"
                    variant="primary"
                    size="large"
                    className="w-full md:w-auto flex items-center gap-2"
                    disabled={uploading}
                  >
                    <FileSpreadsheet className="w-4 h-4" />
                    {uploading ? 'Uploading...' : 'Upload PDF for All Admins'}
                  </Button>
                </form>
              </div>

              <div className="flex flex-col gap-6">
                <div className="card p-5 border border-dark-border/80">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="bg-green-500/10 text-green-400 rounded-full w-10 h-10 flex items-center justify-center">
                      <FileSpreadsheet className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-[0.3em] text-dark-text-secondary">
                        Last Broadcast
                      </p>
                      <p className="text-lg font-semibold text-dark-text-primary">
                        {lastUploadResult ? 'Completed' : 'Not yet run'}
                      </p>
                    </div>
                  </div>

                  {lastUploadResult ? (
                    <>
                      <p className="text-dark-text-secondary text-sm mb-4">
                        {lastUploadResult.createdRecords} admin accounts received this chapter.
                      </p>
                      {lastUploadResult.failures?.length > 0 ? (
                        <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3 text-sm text-red-300 max-h-36 overflow-auto">
                          <p className="font-semibold mb-2">
                            Failed for {lastUploadResult.failures.length} admin(s):
                          </p>
                          <ul className="space-y-1">
                            {lastUploadResult.failures.map((failure, idx) => (
                              <li key={`${failure.admin_id}-${idx}`}>
                                Admin #{failure.admin_id}: {failure.error}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ) : (
                        <div className="bg-green-500/5 border border-green-500/20 rounded-lg p-3 text-sm text-green-300">
                          Distributed to every admin without issues.
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="text-dark-text-secondary text-sm">
                      Once you upload your first PDF, distribution stats will appear here.
                    </p>
                  )}
                </div>

                <div className="card p-5 border border-dark-border/80">
                  <p className="text-xs uppercase tracking-[0.3em] text-dark-text-secondary mb-3">
                    Upload Blueprint
                  </p>
                  <ul className="space-y-3">
                    {uploadGuidelines.map((tip) => (
                      <li key={tip.title} className="flex gap-3">
                        <div className="w-2 h-2 rounded-full bg-purple-400 mt-2.5"></div>
                        <div>
                          <p className="font-semibold text-dark-text-primary text-sm">{tip.title}</p>
                          <p className="text-dark-text-secondary text-sm">{tip.description}</p>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

const FormField = ({ label, hint, value, onChange, required = false }) => (
  <div>
    <label className="text-sm text-dark-text-secondary mb-1 block">
      {label} {required && <span className="text-red-400">*</span>}
    </label>
    <input
      type="text"
      className="input-field bg-dark-bg/40 border border-dark-border rounded-xl px-4 py-2.5 focus:border-purple-500 transition-colors"
      placeholder={hint}
      value={value}
      onChange={onChange}
    />
  </div>
);

export default SuperadministrationDashboard;
